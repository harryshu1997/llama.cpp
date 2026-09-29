"""Automated runtime: phone helpers, assistance fractions and attachment.

Split from test_automated_runtime.py on 2026-09-13; fixtures and the base class stay there."""

from __future__ import annotations

import unittest
from dataclasses import replace
import time
from unittest import mock
from research_dev.scheduler import (
    GGUFModelManifestLoader,
    RuntimeCapabilityCatalog,
    RuntimeCompositeExecutorCapability,
    RuntimeCompositeOperatorPlacement,
    RuntimeDesktopControlProfile,
    RuntimePhonePowerProfile,
)
from research_dev.scheduler._internal.adaptive_decode import ADAPTIVE_OBSERVATION_STORE_SCHEMA
from research_dev.scheduler._internal.adaptive_decode_contracts import (
    AdaptiveDecodeError,
    AdaptiveDecodeGroupedObservation,
    AdaptiveDecodeWindowReceipt,
)
from research_dev.scheduler._internal.adaptive_decode_planning import adaptive_decode_policies
from research_dev.scheduler._internal.types import canonical_sha256
from research_dev.scheduler.adapters import (
    apply_assumed_phone_power_catalog,
    apply_assumed_phone_power_profile,
)

try:
    from .test_automated_runtime import (
        AutomatedRuntimeTests,
        catalog,
        executor_state,
        request,
        runtime_snapshot,
    )
except ImportError:  # run from the tests directory
    from test_automated_runtime import (
        AutomatedRuntimeTests,
        catalog,
        executor_state,
        request,
        runtime_snapshot,
    )


class AutomatedRuntimePhoneTests(AutomatedRuntimeTests):
    """Automated runtime: phone helpers, assistance fractions and attachment."""

    def test_unavailable_phone_epoch_dispatches_desktop(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        first = scheduler.submit_automated_request(
            request("phone-epoch-source"),
            manifest.model_id,
            runtime_snapshot(manifest),
        )
        self.assertIn("helper-c", first.execution_plan.device_ids)
        active = scheduler.wait_runtime_request(
            first.request.request_id,
            time.monotonic_ns() - first.decision.start_us * 1_000,
        )
        scheduler.complete_automated_request(
            first.request.request_id,
            self.execution_receipt(
                active, finished_us=active.decision.start_us + 1
            ),
        )
        observed_at_us = 2_000
        unavailable = runtime_snapshot(manifest, include_phone=False)
        unavailable = replace(
            unavailable,
            snapshot_id="phone-epoch-unavailable",
            captured_at_us=observed_at_us,
            valid_until_us=10_000_000,
            memory=replace(
                unavailable.memory,
                snapshot_id="phone-epoch-unavailable-memory",
                captured_at_us=observed_at_us,
                valid_until_us=10_000_000,
            ),
        )
        fallback = scheduler.submit_automated_request(
            request("phone-epoch-fallback", arrival_us=observed_at_us),
            manifest.model_id,
            unavailable,
            observed_at_us=observed_at_us,
        )

        self.assertNotIn("helper-c", fallback.execution_plan.device_ids)
        self.assertEqual(fallback.dispatch_state, "QUEUED")
        selected = next(
            row for row in fallback.cost_estimates.estimates
            if row.route_id == fallback.decision.route_id
        )
        resolution = selected.details["model_placement_resolution"]
        self.assertEqual(
            resolution["passes"][-1]["live_selected_route_id"],
            fallback.decision.route_id,
        )
        self.assertFalse(any(
            row["event_kind"] in {"FAILED", "CANCELLED"}
            for row in scheduler.runtime_decision_log()["records"]
        ))

    def test_fraction_cost_update_keeps_placement_generation_stable(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        snapshot = runtime_snapshot(manifest)
        ticket = scheduler.submit_automated_request(
            request("fraction-learning-stable", arrival_us=1_000),
            manifest.model_id,
            snapshot,
        )
        epoch = next(iter(
            scheduler._runtime_residency_cohorts
            .compatible_published_model_placement_epochs(
                manifest.artifact_sha256,
                ticket.request.quality_requirement,
                ticket.selection_mode,
                scheduler._runtime_capabilities.maximum_latency_ppm,
            )
        ))
        candidate_set = scheduler._runtime_route_template_sets[
            epoch.epoch_sha256
        ].candidate_set
        before = scheduler._runtime_placement_learning_signature(
            candidate_set
        )
        adjusted = replace(
            candidate_set,
            candidates=tuple(
                replace(
                    row,
                    residency_break_even={
                        **dict(row.residency_break_even or {}),
                        "adaptive_history_selected_fraction_ppm": 750_000,
                    },
                )
                for row in candidate_set.candidates
            ),
        )

        self.assertEqual(
            before,
            scheduler._runtime_placement_learning_signature(adjusted),
        )
        self.assertEqual(
            epoch.learning_generation_sha256,
            scheduler._runtime_placement_learning_generation_sha256(
                manifest.artifact_sha256
            ),
        )

    def test_phone_only_wins_when_fit_qualified_and_energy_positive(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        ticket = scheduler.submit_automated_request(
            request("phone-wins"),
            manifest.model_id,
            runtime_snapshot(manifest, phone_bandwidth=8_000_000_000),
        )
        self.assertEqual(ticket.execution_plan.route_family, "whole_model")
        self.assertEqual(ticket.execution_plan.device_ids, ("helper-c",))
        self.assertEqual(
            ticket.binding.operator_plan_sha256,
            ticket.execution_plan.plan_sha256,
        )
        self.assertEqual(
            [(row.device_id, row.endpoint) for row in ticket.binding.participants],
            [("helper-c", "synthetic://helper-c")],
        )

    def test_assumed_phone_power_requires_explicit_selection_policy(
        self,
    ) -> None:
        source = catalog(
            phone_ops_per_s=12_000_000_000,
            phone_bandwidth=8_000_000_000,
        )
        disabled = RuntimePhonePowerProfile.assumed_4p5w(
            device_id="helper-c",
            domain_id="energy:helper-c",
            allow_assumed_for_scheduling=False,
        )
        disabled_catalog = apply_assumed_phone_power_catalog(
            source, disabled
        )
        scheduler, manifest = self.scheduler_and_manifest(disabled_catalog)
        disabled_ticket = scheduler.submit_automated_request(
            request("assumed-phone-disabled"),
            manifest.model_id,
            runtime_snapshot(manifest, phone_bandwidth=8_000_000_000),
        )
        self.assertNotIn(
            "helper-c", disabled_ticket.execution_plan.device_ids
        )
        self.assertTrue(any(
            "helper-c" in row.details.get("device_ids", [])
            and row.details.get("energy_evidence") == "ASSUMED"
            and row.route_id in dict(disabled_ticket.decision.rejected)
            for row in disabled_ticket.cost_estimates.estimates
        ))

        enabled = replace(disabled, allow_assumed_for_scheduling=True)
        enabled_catalog = apply_assumed_phone_power_catalog(
            source, enabled
        )
        enabled_catalog = RuntimeCapabilityCatalog.from_json(
            enabled_catalog.to_json()
        )
        scheduler, manifest = self.scheduler_and_manifest(enabled_catalog)
        enabled_ticket = scheduler.submit_automated_request(
            request("assumed-phone-enabled"),
            manifest.model_id,
            runtime_snapshot(manifest, phone_bandwidth=8_000_000_000),
        )

        self.assertIn("helper-c", enabled_ticket.execution_plan.device_ids)
        self.assertEqual(
            enabled_ticket.execution_plan.adapter_parameters[
                "phone_power_evidence_kind"
            ],
            "ASSUMED_4P5W",
        )
        selected_cost = next(
            row for row in enabled_ticket.cost_estimates.estimates
            if row.route_id == enabled_ticket.decision.route_id
        )
        self.assertEqual(
            selected_cost.details["energy_evidence"], "ASSUMED"
        )
        decision_record = scheduler.runtime_decision_log()["records"][0]
        selected_record = next(
            row for row in decision_record["candidates"]
            if row["route_id"] == enabled_ticket.decision.route_id
        )
        self.assertEqual(
            selected_record["details"]["phone_power_evidence_kind"],
            "ASSUMED_4P5W",
        )
        self.assertEqual(
            decision_record["selected"]["operator_plan"]
                ["adapter_parameters"]["phone_power_evidence_kind"],
            "ASSUMED_4P5W",
        )

    def test_assumed_phone_power_keeps_low_battery_fail_closed(self) -> None:
        source = catalog(
            phone_ops_per_s=12_000_000_000,
            phone_bandwidth=8_000_000_000,
        )
        phone_power = RuntimePhonePowerProfile.assumed_4p5w(
            device_id="helper-c",
            domain_id="energy:helper-c",
            allow_assumed_for_scheduling=True,
        )
        source = replace(
            source,
            placement_profile=apply_assumed_phone_power_profile(
                source.placement_profile, phone_power
            ),
            executors=tuple(
                replace(row, minimum_battery_ppm=50_000)
                if row.device_id == "helper-c" else row
                for row in source.executors
            ),
            phone_power_profiles=(phone_power,),
        )
        scheduler, manifest = self.scheduler_and_manifest(source)
        snapshot = runtime_snapshot(
            manifest, phone_bandwidth=8_000_000_000
        )
        snapshot = replace(snapshot, executors={
            **snapshot.executors,
            "executor:helper-c": replace(
                snapshot.executors["executor:helper-c"],
                battery_ppm=49_999,
                charging=True,
            ),
        })
        ticket = scheduler.submit_automated_request(
            request("assumed-phone-low-battery"),
            manifest.model_id,
            snapshot,
        )

        self.assertNotIn("helper-c", ticket.execution_plan.device_ids)
        self.assertTrue(any(
            "helper-c" in row.details.get("device_ids", [])
            and row.details.get("primary_rejection_reason")
                == "BATTERY_LIMIT"
            for row in ticket.cost_estimates.estimates
        ))

    def test_phone_power_migration_preserves_only_physical_latency(
        self,
    ) -> None:
        manifest = GGUFModelManifestLoader.load(
            "unseen-model", self.path
        )
        base = catalog(
            phone_ops_per_s=12_000_000_000,
            phone_bandwidth=8_000_000_000,
            phone_whole_model=False,
            gpu_whole_model=False,
        )
        placements = tuple(
            RuntimeCompositeOperatorPlacement(
                operator_id=operator.operator_id,
                primary_device_id=(
                    "host-a"
                    if operator.kind in {"embedding", "kv_cache"}
                    else "accelerator-b"
                ),
                helper_device_id=None,
                split_axis="none",
                split_fraction_ppm=0,
            )
            for operator in manifest.operators
        )
        desktop = RuntimeCompositeExecutorCapability(
            executor_id="coordinator:phone-power-desktop",
            endpoint="synthetic://phone-power-desktop",
            backend="backend:desktop",
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
            layer_fractions_ppm=(),
            residency_states=("hot",),
            resource_ids=(
                "compute:host-a", "compute:accelerator-b",
                "link:pcie-out", "link:pcie-in",
            ),
            operator_plan_protocol="synthetic-desktop-v1",
            maturity="QUALIFIED",
            evidence_ids=("synthetic-phone-power-desktop",),
            artifact_sha256=manifest.artifact_sha256,
            operator_placements=placements,
        )
        phone = RuntimeCompositeExecutorCapability(
            executor_id="coordinator:phone-power-assisted",
            endpoint="synthetic://phone-power-assisted",
            backend="backend:phone",
            coordinator_device_id="host-a",
            participant_device_ids=(
                "host-a", "accelerator-b", "helper-c"
            ),
            participant_resource_ids={
                "host-a": ("compute:host-a",),
                "accelerator-b": ("compute:accelerator-b",),
                "helper-c": ("compute:helper-c",),
            },
            route_family="operator_split",
            assisted_operator_kind="ffn",
            split_axis="column",
            split_fractions_ppm=(250_000, 500_000, 750_000),
            layer_fractions_ppm=(),
            residency_states=("hot",),
            resource_ids=(
                "compute:host-a", "compute:accelerator-b",
                "compute:helper-c", "link:pcie-out", "link:pcie-in",
                "link:usb-out", "link:usb-in",
            ),
            operator_plan_protocol="synthetic-phone-v1",
            maturity="QUALIFIED",
            evidence_ids=("synthetic-phone-power-assisted",),
            artifact_sha256=manifest.artifact_sha256,
            operator_ids=tuple(
                operator.operator_id for operator in manifest.operators
                if operator.kind == "ffn"
            ),
            baseline_executor_id=desktop.executor_id,
            helper_device_id="helper-c",
            adapter_parameters={
                "ffn_column_quantum": 1,
                "ffn_n_embd": manifest.embedding_length,
                "ffn_runtime_control_protocol": "decode-boundary-v1",
                "ffn_weight_buffer_layout": "selected-width",
                "maximum_helper_resident_weight_bytes": (
                    manifest.tensor_bytes
                ),
                "phone_device_id": "helper-c",
                "requires_measured_route_profile": 1,
                "ubatch_size": 4,
            },
        )
        source_catalog = RuntimeCapabilityCatalog.from_json(replace(
            base,
            composite_executors=(desktop, phone),
            desktop_control_profiles=(RuntimeDesktopControlProfile(
                profile_id="synthetic-phone-power-control",
                artifact_sha256=manifest.artifact_sha256,
                executor_id=desktop.executor_id,
                operator_placements=placements,
                maturity="QUALIFIED",
                evidence_ids=("synthetic-phone-power-desktop",),
            ),),
        ).to_json())
        source, manifest = self.scheduler_and_manifest(source_catalog)
        runtime_request = request(
            "phone-power-migration-source",
            input_tokens=33,
            output_tokens=33,
        )
        snapshot = runtime_snapshot(
            manifest, phone_bandwidth=8_000_000_000
        )
        snapshot = replace(
            snapshot,
            executors={
                **snapshot.executors,
                desktop.executor_id: executor_state(desktop.executor_id),
                phone.executor_id: executor_state(phone.executor_id),
            },
        )
        candidates = source.generate_automated_candidates(
            runtime_request, manifest.model_id, snapshot
        )
        baseline, policies, envelope = adaptive_decode_policies(
            candidates,
            manifest,
            source._runtime_capabilities,
            runtime_request.output_tokens,
        )
        self.assertIsNotNone(envelope)
        selected = max(policies, key=lambda row: row.split_fraction_ppm)
        groups = []
        for index in range(2):
            request_id = "phone-power-migration-" + str(index)
            baseline_window = AdaptiveDecodeWindowReceipt(
                request_id=request_id,
                slot_id=0,
                window_index=0,
                token_start=0,
                token_end=4,
                context_length=runtime_request.input_tokens,
                active_batch=1,
                started_at_us=0,
                finished_at_us=4_000,
                policy=baseline,
                applied_ack=None,
                fleet_energy_uj_by_domain={
                    "cpu-package": 100_000,
                    "energy:helper-c": 1_000,
                    "gpu-board": 50_000,
                },
                latency_per_token_us=1_000,
                phone_compute_us=0,
                usb_transfer_us=0,
                rpc_us=0,
                exposed_tail_us=0,
                output_valid=True,
                evidence_ids=("legacy-measured-phone-energy",),
                energy_boundary_id=(
                    source_catalog.placement_profile.energy_boundary_id
                ),
                energy_attribution_kind="isolated",
                failure_reason=None,
                previous_record_sha256="0" * 64,
            )
            assisted_window = AdaptiveDecodeWindowReceipt(
                request_id=request_id,
                slot_id=0,
                window_index=1,
                token_start=4,
                token_end=8,
                context_length=runtime_request.input_tokens,
                active_batch=1,
                started_at_us=4_000,
                finished_at_us=7_200,
                policy=selected,
                applied_ack=None,
                fleet_energy_uj_by_domain={
                    "cpu-package": 40_000,
                    "energy:helper-c": 5_000,
                    "gpu-board": 20_000,
                },
                latency_per_token_us=800,
                phone_compute_us=2_000,
                usb_transfer_us=800,
                rpc_us=100,
                exposed_tail_us=300,
                output_valid=True,
                evidence_ids=("legacy-measured-phone-energy",),
                energy_boundary_id=(
                    source_catalog.placement_profile.energy_boundary_id
                ),
                energy_attribution_kind="isolated",
                failure_reason=None,
                previous_record_sha256=(
                    baseline_window.record_sha256.removeprefix("sha256:")
                ),
                usb_upload_bytes=7_680,
                usb_download_bytes=7_680,
                completed_phone_calls=4,
                completed_phone_input_rows=4,
            )
            groups.append(AdaptiveDecodeGroupedObservation(
                request_id=request_id,
                ticket_id=request_id + ":attempt:0",
                model_artifact_sha256=manifest.artifact_sha256,
                planning_profile_sha256=(
                    source._runtime_capability_generation_sha256
                ),
                desktop_placement_sha256=(
                    baseline.desktop_placement_sha256
                ),
                windows=(baseline_window, assisted_window),
                final_policy=selected,
                terminal_status="COMPLETED",
                state_history=(
                    "BASELINE", "PROBING", "EXPLOITING", "COMPLETED"
                ),
            ))
        body = {
            "groups": [row.to_json() for row in groups],
            "schema": ADAPTIVE_OBSERVATION_STORE_SCHEMA,
        }
        phone_power = RuntimePhonePowerProfile.assumed_4p5w(
            device_id="helper-c",
            domain_id="energy:helper-c",
            allow_assumed_for_scheduling=True,
        )
        target_catalog = apply_assumed_phone_power_catalog(
            source_catalog, phone_power
        )
        self.assertEqual(
            target_catalog.placement_profile,
            source_catalog.placement_profile,
        )
        target, target_manifest = self.scheduler_and_manifest(
            target_catalog
        )
        target.load_adaptive_decode_observations(
            {
                **body,
                "store_sha256": canonical_sha256(body),
            },
            source_catalog=source_catalog,
        )

        restored = target.generate_automated_candidates(
            replace(runtime_request, request_id="phone-power-migration-target"),
            target_manifest.model_id,
            snapshot,
        )
        migrated = tuple(
            row for row in restored.candidates
            if (row.residency_break_even or {}).get(
                "adaptive_history_selected_group_count"
            ) == 2
        )
        self.assertTrue(migrated)
        self.assertTrue(all(
            row.maturity == "QUALIFIED"
            and row.cost.latency_evidence == "MEASURED"
            and row.cost.energy_evidence == "ASSUMED"
            and (row.residency_break_even or {}).get(
                "phone_energy_evidence"
            ) == "ASSUMED_4P5W"
            and (row.residency_break_even or {}).get(
                "paired_energy_evidence"
            ) == "ASSUMED_4P5W"
            and "ROUTE_NOT_QUALIFIED" not in row.rejection_reasons
            for row in migrated
        ))
        desktop_ticket = target.submit_automated_request(
            replace(
                runtime_request,
                request_id="phone-power-migration-desktop",
            ),
            target_manifest.model_id,
            snapshot,
            selection_mode="desktop-baseline",
        )
        self.assertNotIn(
            "helper-c", desktop_ticket.execution_plan.device_ids
        )
        self.assertEqual(
            desktop_ticket.decision.route_id,
            restored.baseline.candidate_id,
        )
        target._legacy_evidence_migration_cache.clear()
        with mock.patch.object(
            target._adaptive_decode,
            "rebind_legacy_component_observations",
            side_effect=AdaptiveDecodeError(
                "adaptive evidence cannot rebind while active"
            ),
        ) as rebind:
            deferred = target.generate_automated_candidates(
                replace(
                    runtime_request,
                    request_id="phone-power-migration-active",
                    input_tokens=runtime_request.input_tokens + 1,
                ),
                target_manifest.model_id,
                snapshot,
            )
        self.assertTrue(rebind.called)
        self.assertEqual(
            deferred.baseline.candidate_id,
            restored.baseline.candidate_id,
        )

    def test_split_fraction_changes_with_queue_and_bandwidth(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=2_000_000_000,
            phone_power_mw=2_000,
            phone_bandwidth=4_000_000_000,
            phone_whole_model=False,
        ))
        fast_link = scheduler.generate_automated_candidates(
            request("split-fast"),
            manifest.model_id,
            runtime_snapshot(manifest, phone_bandwidth=4_000_000_000),
        )
        slow_request = replace(
            request("split-slow"), deadline_us=2_001_300
        )
        slow_link_busy_gpu = scheduler.generate_automated_candidates(
            slow_request,
            manifest.model_id,
            runtime_snapshot(
                manifest,
                phone_bandwidth=2_000_000,
                gpu_busy_until_us=2_000_000,
            ),
        )

        def best_fraction(rows, devices):
            candidates = [
                row for row in rows.candidates
                if row.route_family == "operator_split"
                and row.assisted_operator_kind == "ffn"
                and set(row.device_ids) == set(devices)
                and row.split_axis == "column"
                and row.admitted
                and not row.pareto_dominated
            ]
            self.assertTrue(candidates)
            return min(
                candidates,
                key=lambda row: (
                    row.cost.fleet_energy_upper_uj,
                    row.cost.finish_upper_us,
                    row.candidate_id,
                ),
            ).split_fraction_ppm

        cpu_phone = best_fraction(fast_link, ("host-a", "helper-c"))
        gpu_phone = best_fraction(
            slow_link_busy_gpu, ("accelerator-b", "helper-c")
        )
        self.assertNotEqual(cpu_phone, gpu_phone)

    def test_operator_specialist_phone_needs_only_its_declared_kernel(self) -> None:
        base_catalog = catalog(phone_whole_model=False)
        phone = replace(
            base_catalog.executor_by_device["helper-c"],
            kernel_profiles={"ffn": "kernel:helper-c:ffn"},
            supports_layer_placement=False,
            supports_kv_cache=False,
        )
        specialist_catalog = replace(
            base_catalog,
            executors=tuple(
                phone if row.device_id == "helper-c" else row
                for row in base_catalog.executors
            ),
        )
        scheduler, manifest = self.scheduler_and_manifest(specialist_catalog)
        candidates = scheduler.generate_automated_candidates(
            request("specialist-phone"),
            manifest.model_id,
            runtime_snapshot(manifest),
        )
        assisted = [
            row for row in candidates.candidates
            if "helper-c" in row.device_ids
            and row.route_family in {"operator_offload", "operator_split"}
        ]
        self.assertTrue(assisted)
        self.assertEqual(
            {row.assisted_operator_kind for row in assisted}, {"ffn"}
        )
        self.assertFalse(any(
            row.route_family in {"whole_model", "layer_placement"}
            and "helper-c" in row.device_ids
            for row in candidates.candidates
        ))


if __name__ == "__main__":
    unittest.main()
