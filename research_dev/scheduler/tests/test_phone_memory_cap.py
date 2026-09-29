from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from research_dev.scheduler import (
    DeviceMemoryCapacity, RuntimeLinkState, RuntimeMemoryDemand, UnifiedScheduler,
)
from research_dev.scheduler._internal.runtime_capabilities import RuntimePhonePowerProfile
from research_dev.scheduler.adapters import materialize_whole_model_endpoint
from research_dev.scheduler.adapters.snapshot import (
    EndpointRuntimeSample, UnifiedRuntimeSnapshotBuilder, live_executor_residency_sample,
)
from research_dev.scheduler._internal.phone_shards import (
    PhoneFfnResidencyDemand, PhoneFfnShardStorageMetadata, generate_mixed_ffn_residency_layouts,
    progressive_ffn_residency_layouts,
)
from research_dev.scheduler._internal.route_generation.feasibility import RouteFeasibilityMixin
from research_dev.scheduler._internal.model_placement_controller import ModelPlacementController
from research_dev.scheduler._unified.phone_residency_ops.economics import (
    _memory_cap_candidate_choice, _phone_cap_needs_reevaluation, _phone_memory_budget,
)
from research_dev.scheduler._unified.phone_residency_ops.publication import _phone_service_admission_reason
from research_dev.scheduler._unified.placement_epochs_ops.frontier import (
    _persistent_phone_service_reserve_by_artifact,
)
from research_dev.scheduler._unified.automated_requests_ops.selection import _submit_candidate_set
from research_dev.scheduler._unified.automated_requests_ops.replan_selection import _replan_candidate_set
from research_dev.scheduler._unified.automated_selection_ops.materialization import (
    _rematerialize_epoch_candidates,
)

try:
    from .test_multi_session_phone import manifest, session
    from .test_automated_runtime import catalog, runtime_snapshot, request
except ImportError:
    from test_multi_session_phone import manifest, session
    from test_automated_runtime import catalog, runtime_snapshot, request


class PhoneMemoryCapTests(unittest.TestCase):
    def setUp(self):
        self.model = manifest(6)
        self.sessions = tuple(session(index) for index in range(3))
        self.demand = PhoneFfnResidencyDemand(self.model, 100, 8, "split-row")
        layouts = generate_mixed_ffn_residency_layouts(
            (self.demand,), self.sessions, phone_wide_limit_bytes=768,
        )
        self.source = max(layouts, key=lambda row: row.resident_bytes).with_session_generations(
            {row.session_id: 1 for row in self.sessions}
        )

    def resized(self, limit, *, demand=True):
        return generate_mixed_ffn_residency_layouts(
            (self.demand,) if demand else (), self.sessions,
            phone_wide_limit_bytes=limit, current_shards=self.source.shards,
            resident_manifests=(self.model,),
            allow_resident_shrink=True,
        )

    def controller(self, *, peak=320):
        whole = replace(manifest(1), artifact_sha256="sha256:" + "d" * 64,
                        model_id="synthetic-whole-service")
        parameters = {"persistent_residency": 1, "execution_adapter": "android-llama-server-v1",
                      "gpu_device_id": "phone-a"}
        if peak is not None:
            parameters["whole_model_peak_memory_bytes"] = peak
        helper = SimpleNamespace(
            device_id="phone-a", executor_id="whole-service", memory_resource_id="phone-memory",
            phone_sessions=self.sessions, adapter_parameters=parameters,
        )
        catalog = SimpleNamespace(
            executor_by_device={"phone-a": helper}, executors=(helper,),
            transitions=(SimpleNamespace(executor_id="whole-service", artifact_sha256=whole.artifact_sha256,
                                         target_state="hot", energy_maturity="QUALIFIED"),),
            placement_profile=SimpleNamespace(
                devices={"phone-a": SimpleNamespace(allocation_limit_bytes=1000)},
                memory_pools={"phone-memory": SimpleNamespace(capacity_bytes=1000, reserved_bytes=0)},
            ),
        )
        owner = SimpleNamespace(
            ready=SimpleNamespace(layout=self.source, workspace_bytes=32),
            policy=SimpleNamespace(phone_minimum_residency_us=30_000_000),
        )
        owner.ready_phone_layout = lambda: owner.ready
        owner.planning_phone_layout = lambda: owner.ready
        controller = SimpleNamespace(
            _runtime_capabilities=catalog,
            _runtime_manifests={self.model.model_id: self.model, whole.model_id: whole},
            _model_placement_controller=owner,
            _phone_htp_memory_caps={"phone-a": (1000, 32)},
            _phone_layout_request_impacts=lambda *args: (),
        )
        controller._persistent_phone_service_reserve_by_artifact = (
            lambda device, snapshot, **kwargs: _persistent_phone_service_reserve_by_artifact(
                controller, device, snapshot, **kwargs,
            )
        )
        controller._phone_memory_budget = lambda device, sessions, snapshot: _phone_memory_budget(
            controller, device, sessions, snapshot,
        )
        return controller, whole, helper

    def snapshot(self, occupied=768, residency=()):
        return SimpleNamespace(
            captured_at_us=1,
            memory=SimpleNamespace(capacities={
                "phone-memory": DeviceMemoryCapacity("phone-memory", 1000, occupied, 0),
            }),
            residency=residency, telemetry_observations={"phone-a": {"validity": "VALID"}},
            telemetry_unavailable_reason=lambda *args: None,
        )

    def test_smaller_cap_keeps_three_nonempty_sessions(self):
        before = self.source.to_json()
        layouts = [row for row in self.resized(640) if len(row.shards) == 3]
        self.assertTrue(layouts)
        target = min(layouts, key=lambda row: (len(row.changed_session_ids), row.objective))
        self.assertEqual(len(target.changed_session_ids), 1)
        selected = target.changed_session_ids[0]
        source = {row.session_id: row for row in self.source.shards}
        for row in target.shards:
            self.assertGreater(row.resident_bytes, 0)
            self.assertEqual(row.artifact_sha256, source[row.session_id].artifact_sha256)
            self.assertEqual(row.layer_mask & ~source[row.session_id].layer_mask, 0)
            if row.session_id != selected:
                self.assertEqual(row, source[row.session_id])
        self.assertLessEqual(target.resident_bytes, 640)
        self.assertEqual(self.source.to_json(), before)
        self.assertEqual(set(self.source.session_generation_by_id.values()), {1})

    def test_resize_does_not_invent_demand_for_idle_artifacts(self):
        target = min(self.resized(640, demand=False), key=lambda row: (
            len(row.changed_session_ids), -row.resident_bytes,
        ))
        self.assertEqual(len(target.shards), 3)
        self.assertEqual(dict(target.queued_work_by_artifact), {})
        self.assertEqual(target.queue_benefit, 0)

    def test_resize_is_not_enabled_for_existing_layout_selection(self):
        layouts = generate_mixed_ffn_residency_layouts(
            (), self.sessions, phone_wide_limit_bytes=640,
            current_shards=self.source.shards, resident_manifests=(self.model,),
        )
        self.assertEqual(layouts, ())

    def test_resize_rejects_unsupported_resident_dtype(self):
        sessions = tuple(replace(row, supported_data_types=("Q8_0",)) for row in self.sessions)
        layouts = generate_mixed_ffn_residency_layouts(
            (), sessions, phone_wide_limit_bytes=640,
            current_shards=self.source.shards, resident_manifests=(self.model,),
            allow_resident_shrink=True,
        )
        self.assertEqual(layouts, ())

    def _regrowth(self, *, limit=768, allowed=None, storage=(), benefit=1_000_000,
                  queued_work=100, enabled=True, sessions=None):
        shrunk = min((row for row in self.resized(640) if len(row.shards) == 3),
                     key=lambda row: (len(row.changed_session_ids), -row.resident_bytes))
        shrunk = shrunk.with_session_generations({
            row.session_id: 1 + int(row.session_id in shrunk.changed_session_ids) for row in shrunk.shards
        })
        demand = replace(
            self.demand, allowed_operator_ids=allowed, queued_work=max(1, queued_work),
            benefit_by_operator={row.operator_id: benefit for row in self.model.operators},
            benefit_value_kind="measured_net_energy_uj",
        )
        layouts = generate_mixed_ffn_residency_layouts(
            (demand,) if queued_work else (), self.sessions if sessions is None else sessions,
            phone_wide_limit_bytes=limit,
            current_shards=shrunk.shards, resident_manifests=(self.model,),
            shard_storage=storage, allow_resident_shrink=enabled,
            transition_energy_uj_by_session={row.session_id: 1000 for row in self.sessions},
        )
        return shrunk, layouts

    def test_cap_increase_can_regrow_one_same_artifact_session(self):
        shrunk, layouts = self._regrowth()
        before = shrunk.to_json()
        grown = [row for row in layouts if row.resident_bytes > shrunk.resident_bytes]
        self.assertTrue(grown)
        target = min(grown, key=lambda row: row.objective)
        self.assertEqual(target.resident_bytes, 768)
        selected, = target.changed_session_ids
        previous = {row.session_id: row for row in shrunk.shards}
        masks = 0
        for row in target.shards:
            self.assertEqual(row.artifact_sha256, previous[row.session_id].artifact_sha256)
            self.assertEqual(masks & row.layer_mask, 0)
            masks |= row.layer_mask
            if row.session_id != selected:
                self.assertEqual(row, previous[row.session_id])
            else:
                self.assertEqual(previous[row.session_id].layer_mask & ~row.layer_mask, 0)
        self.assertEqual(shrunk.to_json(), before)
        self.assertEqual(shrunk.session_generation_by_id[selected], 2)
        self.assertTrue(any(row.geometry_sha256 == shrunk.geometry_sha256 for row in layouts))

    def test_regrowth_keeps_cpu_parent_and_storage_coverage(self):
        shrunk, _ = self._regrowth()
        resident_ids = tuple(key for shard in shrunk.shards for key in shard.operator_ids)
        stored = tuple(PhoneFfnShardStorageMetadata(
            row.artifact_sha256, "sha256:" + "e" * 64, "/staged/" + row.session_id,
            row.layer_mask, row.maximum_columns, row.session_id,
        ) for row in shrunk.shards)
        for options in ({"allowed": resident_ids}, {"storage": stored}, {"limit": 700},
                        {"queued_work": 0}, {"enabled": False}):
            with self.subTest(options=options):
                _, layouts = self._regrowth(**options)
                self.assertFalse(any(row.resident_bytes > shrunk.resident_bytes for row in layouts))

    def test_regrowth_is_still_subject_to_transition_economics(self):
        shrunk, layouts = self._regrowth(benefit=1, queued_work=1)
        grown = [row for row in layouts if row.resident_bytes > shrunk.resident_bytes]
        self.assertTrue(grown)
        retained = next(row for row in layouts if row.geometry_sha256 == shrunk.geometry_sha256)
        self.assertLess(retained.objective, min(row.objective for row in grown))
        selected, _, _ = ModelPlacementController().select_phone_layout_candidate(
            layouts, current_layout=shrunk, minimum_energy_saving_ppm=0,
        )
        self.assertEqual(selected.geometry_sha256, shrunk.geometry_sha256)

    def test_regrowth_does_not_keep_an_invalid_retained_session(self):
        shrunk, _ = self._regrowth()
        retained_id = next(row.session_id for row in shrunk.shards
                           if row.session_id not in shrunk.changed_session_ids)
        for changes in ({"resident_memory_limit_bytes": 128}, {"ready": False},
                        {"endpoint": "session://replacement-worker"}):
            with self.subTest(changes=changes):
                sessions = tuple(replace(row, **changes) if row.session_id == retained_id else row
                                 for row in self.sessions)
                _, layouts = self._regrowth(sessions=sessions)
                self.assertFalse(any(len(row.changed_session_ids) == 1
                                     and row.resident_bytes > shrunk.resident_bytes for row in layouts))

    def test_cap_growth_is_reconsidered_without_a_new_arrival(self):
        controller, _, _ = self.controller(peak=128)
        shrunk, _ = self._regrowth()
        owner = controller._model_placement_controller
        owner.ready.layout = shrunk
        previous = {"kind": "EVALUATED", "phone_memory_selected_limit_bytes": 640,
                    "current_geometry_sha256": shrunk.geometry_sha256}
        owner.phone_layout_events = lambda: (previous,)
        snapshot = self.snapshot(672)
        self.assertTrue(_phone_cap_needs_reevaluation(controller, snapshot))
        previous["phone_memory_selected_limit_bytes"] = 768
        self.assertFalse(_phone_cap_needs_reevaluation(controller, snapshot))
        controller._phone_htp_memory_caps["phone-a"] = (600, 32)
        self.assertTrue(_phone_cap_needs_reevaluation(controller, snapshot))

    def test_multiple_resizes_still_publish_one_session_per_stage(self):
        target = next(row for row in self.resized(384) if len(row.shards) == 3)
        stages = progressive_ffn_residency_layouts(target, current_shards=self.source.shards)
        previous = {row.session_id: row for row in self.source.shards}
        self.assertEqual(len(stages), 3)
        for stage in stages:
            selected, = stage.changed_session_ids
            for row in stage.shards:
                if row.session_id != selected:
                    self.assertEqual(row, previous[row.session_id])
            self.assertLess(stage.resident_bytes, sum(row.resident_bytes for row in previous.values()))
            previous = {row.session_id: row for row in stage.shards}

    def test_cap_that_cannot_keep_sessions_nonempty_defers(self):
        controller, _, _ = self.controller()
        memory = controller._phone_memory_budget("phone-a", self.sessions, self.snapshot())
        memory = replace(memory, phone_wide_limit=128)
        choice = _memory_cap_candidate_choice(controller, self.resized(128), memory, {}, 1)
        self.assertEqual(choice.reason, "PHONE_RESIDENCY_MEMORY_CAP_DEFERRED")
        self.assertIs(choice.selected, self.source)
        self.assertFalse(choice.force)

    def test_cap_selects_resize_without_claiming_ready_memory(self):
        controller, _, _ = self.controller()
        memory = controller._phone_memory_budget("phone-a", self.sessions, self.snapshot())
        choice = _memory_cap_candidate_choice(controller, self.resized(648), memory, {}, 1)
        self.assertEqual(choice.reason, "PHONE_RESIDENCY_MEMORY_CAP_REBALANCE")
        self.assertEqual(len(choice.selected.changed_session_ids), 1)
        self.assertLessEqual(choice.selected.resident_bytes, 648)
        self.assertIs(controller._model_placement_controller.ready.layout, self.source)

    def test_peak_reservation_includes_workspace_and_shrinks_weight_budget(self):
        controller, whole, _ = self.controller()
        memory = controller._phone_memory_budget("phone-a", self.sessions, self.snapshot())
        self.assertEqual(memory.persistent_service_reserve_bytes, 320)
        self.assertEqual(memory.persistent_service_peak_by_artifact[whole.artifact_sha256], 320)
        self.assertEqual(memory.phone_wide_limit, 648)

    def test_live_resident_workspace_is_counted_once(self):
        controller, whole, helper = self.controller(peak=192)
        candidate = SimpleNamespace(plan=SimpleNamespace(adapter_parameters=helper.adapter_parameters))
        observed = SimpleNamespace(
            device_id="phone-a", artifact_sha256=whole.artifact_sha256,
            state="hot", executor_id="whole-service", resident_bytes=128,
            reclaimable_bytes=192,
        )
        before = self.source.to_json()
        for occupied, residency in ((800, ()), (992, (observed,))):
            with self.subTest(occupied=occupied):
                snapshot = self.snapshot(occupied, residency)
                memory = controller._phone_memory_budget("phone-a", self.sessions, snapshot)
                self.assertEqual(memory.live_phone_wide_limit, 776)
                self.assertEqual(memory.phone_wide_limit, 768)
                self.assertEqual(memory.htp_workspace_bytes, 32)
                self.assertIs(memory.current, self.source)
                self.assertIsNone(_phone_service_admission_reason(controller, candidate, whole, snapshot))
        self.assertEqual(self.source.to_json(), before)

    def test_live_budget_reserves_only_incremental_workspace_growth(self):
        controller, _, _ = self.controller(peak=192)
        controller._phone_htp_memory_caps["phone-a"] = (1000, 64)
        memory = controller._phone_memory_budget("phone-a", self.sessions, self.snapshot(800))
        self.assertEqual(memory.htp_workspace_bytes, 64)
        self.assertEqual(memory.live_phone_wide_limit, 744)
        self.assertEqual(memory.phone_wide_limit, 744)
        self.assertEqual(controller._model_placement_controller.ready.workspace_bytes, 32)

    def test_proposed_workspace_does_not_receive_resident_credit(self):
        controller, _, _ = self.controller(peak=192)
        owner = controller._model_placement_controller
        proposed = owner.ready
        owner.ready = None
        owner.planning_phone_layout = lambda: proposed
        memory = controller._phone_memory_budget("phone-a", self.sessions, self.snapshot(16))
        self.assertEqual(memory.accepted_resident_bytes, 0)
        self.assertEqual(memory.live_phone_wide_limit, 760)
        self.assertEqual(memory.phone_wide_limit, 760)

    def test_workspace_credit_does_not_hide_other_resident_memory(self):
        controller, whole, helper = self.controller(peak=192)
        snapshot = self.snapshot(850)
        memory = controller._phone_memory_budget("phone-a", self.sessions, snapshot)
        self.assertEqual(memory.phone_wide_limit, 726)
        candidate = SimpleNamespace(plan=SimpleNamespace(adapter_parameters=helper.adapter_parameters))
        self.assertEqual(_phone_service_admission_reason(controller, candidate, whole, snapshot),
                         "PHONE_SERVICE_MEMORY_REBALANCE_PENDING")

    def test_capped_static_budget_keeps_live_reserve_with_larger_physical_ram(self):
        controller, _, _ = self.controller(peak=192)
        snapshot = self.snapshot(800)
        snapshot.memory.capacities["phone-memory"] = DeviceMemoryCapacity("phone-memory", 1600, 800, 128)
        memory = controller._phone_memory_budget("phone-a", self.sessions, snapshot)
        self.assertEqual(memory.phone_wide_limit, 1000 - 128 - 192 - 32)
        self.assertEqual(memory.persistent_service_reserve_bytes, 192)

    def test_observed_service_is_not_reserved_twice(self):
        controller, whole, _ = self.controller()
        controller._model_placement_controller.ready.layout = next(
            row for row in self.resized(640) if row.resident_bytes == 640 and len(row.shards) == 3
        )
        before = controller._phone_memory_budget("phone-a", self.sessions, self.snapshot(640))
        observed = SimpleNamespace(device_id="phone-a", artifact_sha256=whole.artifact_sha256,
                                   state="hot", executor_id="whole-service", resident_bytes=128,
                                   reclaimable_bytes=320)
        after = controller._phone_memory_budget("phone-a", self.sessions, self.snapshot(960, (observed,)))
        self.assertEqual(after.persistent_service_reserve_bytes, 0)
        self.assertEqual(after.phone_wide_limit, before.phone_wide_limit)
        self.assertEqual(after.persistent_service_peak_by_artifact, before.persistent_service_peak_by_artifact)

    def test_unknown_service_peak_does_not_assume_weights_are_the_total(self):
        controller, _, _ = self.controller(peak=None)
        memory = controller._phone_memory_budget("phone-a", self.sessions, self.snapshot())
        self.assertEqual(memory.reservation_error, "PHONE_SERVICE_PEAK_MEMORY_UNKNOWN")
        self.assertEqual(memory.phone_wide_limit, 0)

    def test_whole_service_waits_for_actual_resize_not_target_projection(self):
        controller, whole, helper = self.controller()
        candidate = SimpleNamespace(plan=SimpleNamespace(adapter_parameters=helper.adapter_parameters))
        self.assertEqual(_phone_service_admission_reason(controller, candidate, whole, self.snapshot()),
                         "PHONE_SERVICE_MEMORY_REBALANCE_PENDING")
        controller._model_placement_controller.ready.layout = next(
            row for row in self.resized(640) if row.resident_bytes == 640 and len(row.shards) == 3
        )
        self.assertIsNone(_phone_service_admission_reason(controller, candidate, whole, self.snapshot(640)))

    def test_missing_phone_observation_cannot_use_another_device_sample(self):
        controller, whole, helper = self.controller()
        candidate = SimpleNamespace(plan=SimpleNamespace(adapter_parameters=helper.adapter_parameters))
        snapshot = self.snapshot()
        snapshot.telemetry_observations = {"unrelated-device": {"validity": "VALID"}}
        self.assertEqual(_phone_service_admission_reason(controller, candidate, whole, snapshot),
                         "PHONE_SERVICE_MEMORY_OBSERVATION_UNAVAILABLE")

    def test_observation_expiry_is_checked_at_admission_time(self):
        controller, whole, helper = self.controller()
        candidate = SimpleNamespace(plan=SimpleNamespace(adapter_parameters=helper.adapter_parameters))
        snapshot = self.snapshot()
        snapshot.telemetry_unavailable_reason = lambda device, at: "STALE" if at > 10 else None
        self.assertEqual(_phone_service_admission_reason(controller, candidate, whole, snapshot, 11),
                         "PHONE_SERVICE_MEMORY_OBSERVATION_UNAVAILABLE")

    def _cached_admission_reason(self, path, validity):
        controller, whole, helper = self.controller(peak=192)
        snapshot = self.snapshot(800)
        snapshot.cost_features = {}
        observed_at_us = 11
        if validity == "missing":
            snapshot.telemetry_observations = {}
        elif validity == "stale":
            snapshot.telemetry_unavailable_reason = lambda device, at: "STALE" if at > 10 else None
        request = SimpleNamespace(request_id="whole-request", output_tokens=64)
        candidate = SimpleNamespace(plan=SimpleNamespace(adapter_parameters=helper.adapter_parameters))
        candidates = object()
        epoch = SimpleNamespace(epoch_sha256="cached-epoch", generation=1, virtual_queue_request_count=1)
        templates = object()
        compiler = Mock()
        compiler.reuse_exact_route_template_set.return_value = candidates
        compiler.materialize_route_template_set.return_value = candidates
        controller._runtime_residency_cohorts = SimpleNamespace(holds=Mock(return_value={}))
        controller._runtime_controller = SimpleNamespace(current_tickets=Mock(return_value=()))
        controller._prepare_legacy_evidence_migrations = Mock()
        controller._apply_adaptive_history_costs = lambda result, *args: result
        controller._update_phone_residency_portfolio = Mock(return_value=False)
        controller._candidate_set_with_epoch = lambda result, *args, **kwargs: result
        controller._authorize_epoch_compatible_routes = lambda result, *args: result
        controller._route_template_phone_helper_route_ids = Mock(return_value=())
        controller._generate_automated_candidate_set = Mock(side_effect=AssertionError("cache was bypassed"))
        controller._apply_phone_residency_portfolio_authorization = Mock(side_effect=(
            lambda result, model, request, **kwargs: _phone_service_admission_reason(
                controller, candidate, model, kwargs.get("snapshot"), kwargs.get("observed_at_us"),
            )
        ))
        context = SimpleNamespace(
            snapshot=snapshot, placement_epoch=epoch, route_templates=templates,
            epoch_invalidation_reason="", timings_ns={"epoch_validation": 0, "generation": 0},
            compiler=compiler, request=request, manifest=whole, observed_at_us=observed_at_us,
            residency_holds={},
        )
        if path == "selection":
            reason = _submit_candidate_set(
                controller, context=context, request=request, manifest=whole, compiler=compiler,
                observed_at_us=observed_at_us, selection_mode="enforce",
            )
        elif path == "replan":
            reason = _replan_candidate_set(
                controller,
                preparation=SimpleNamespace(context=context, prioritize_resident_component=False,
                                            residency_holds={}),
                current=SimpleNamespace(request=request, selection_mode="enforce"),
                manifest=whole, compiler=compiler, observed_at_us=observed_at_us,
            )
        else:
            reason = _rematerialize_epoch_candidates(controller, context, epoch, templates)
        expected = None if validity == "valid" else "PHONE_SERVICE_MEMORY_OBSERVATION_UNAVAILABLE"
        self.assertEqual(reason, expected)
        controller._apply_phone_residency_portfolio_authorization.assert_called_once_with(
            candidates, whole, request, snapshot=snapshot, observed_at_us=observed_at_us,
        )
        controller._generate_automated_candidate_set.assert_not_called()

    def test_cached_selection_uses_live_admission_observation(self):
        for validity in ("valid", "missing", "stale"):
            with self.subTest(validity=validity):
                self._cached_admission_reason("selection", validity)

    def test_cached_replan_uses_live_admission_observation(self):
        for validity in ("valid", "missing", "stale"):
            with self.subTest(validity=validity):
                self._cached_admission_reason("replan", validity)

    def test_epoch_rematerialization_uses_live_admission_observation(self):
        for validity in ("valid", "missing", "stale"):
            with self.subTest(validity=validity):
                self._cached_admission_reason("epoch", validity)

    def test_wrong_executor_does_not_credit_whole_service_residency(self):
        controller, whole, _ = self.controller()
        observed = SimpleNamespace(device_id="phone-a", artifact_sha256=whole.artifact_sha256,
                                   state="hot", executor_id="different-service", resident_bytes=128,
                                   reclaimable_bytes=320)
        reserved = controller._persistent_phone_service_reserve_by_artifact(
            "phone-a", self.snapshot(640, (observed,)),
        )
        self.assertEqual(reserved[whole.artifact_sha256], 320)

    def test_observed_unqualified_service_still_reserves_its_peak(self):
        controller, whole, helper = self.controller()
        controller._runtime_capabilities.transitions[0].energy_maturity = "SHADOW"
        observed = SimpleNamespace(device_id="phone-a", artifact_sha256=whole.artifact_sha256,
                                   state="hot", executor_id=helper.executor_id, resident_bytes=128,
                                   reclaimable_bytes=192)
        snapshot = self.snapshot(960, (observed,))
        self.assertEqual(controller._persistent_phone_service_reserve_by_artifact(
            "phone-a", snapshot), {whole.artifact_sha256: 128})
        self.assertEqual(controller._persistent_phone_service_reserve_by_artifact(
            "phone-a", snapshot, include_observed=True), {whole.artifact_sha256: 320})

    def test_cold_unqualified_service_with_cap_requires_real_space_not_energy_maturity(self):
        controller, whole, helper = self.controller(peak=192)
        controller._runtime_capabilities.transitions[0].energy_maturity = "SHADOW"
        candidate = SimpleNamespace(plan=SimpleNamespace(adapter_parameters=helper.adapter_parameters))
        self.assertIsNone(_phone_service_admission_reason(controller, candidate, whole, self.snapshot(800)))
        self.assertEqual(controller._runtime_capabilities.transitions[0].energy_maturity, "SHADOW")
        self.assertEqual(_phone_service_admission_reason(controller, candidate, whole, self.snapshot(850)),
                         "MEMORY_CAPACITY")
        helper.adapter_parameters["whole_model_peak_memory_bytes"] = 320
        self.assertEqual(_phone_service_admission_reason(controller, candidate, whole, self.snapshot(800)),
                         "PHONE_SERVICE_MEMORY_REBALANCE_PENDING")

    def test_implicit_cap_still_enforces_declared_shared_pool(self):
        controller, whole, helper = self.controller()
        controller._phone_htp_memory_caps.clear()
        candidate = SimpleNamespace(plan=SimpleNamespace(adapter_parameters=helper.adapter_parameters))
        snapshot = self.snapshot()
        snapshot.memory.capacities["phone-memory"] = DeviceMemoryCapacity("phone-memory", 2000, 800, 0)
        self.assertEqual(_phone_service_admission_reason(controller, candidate, whole, snapshot),
                         "PHONE_SERVICE_MEMORY_REBALANCE_PENDING")

    def test_implicit_cap_accepts_capacity_valid_portfolio_without_double_count(self):
        controller, whole, helper = self.controller(peak=192)
        controller._phone_htp_memory_caps.clear()
        candidate = SimpleNamespace(plan=SimpleNamespace(adapter_parameters=helper.adapter_parameters))
        self.assertIsNone(_phone_service_admission_reason(controller, candidate, whole, self.snapshot(800)))

    def test_implicit_cap_does_not_credit_an_unpublished_resize(self):
        controller, whole, helper = self.controller()
        controller._phone_htp_memory_caps.clear()
        owner = controller._model_placement_controller
        target = min(self.resized(640), key=lambda row: row.objective)
        owner.planning_phone_layout = lambda: SimpleNamespace(layout=target, workspace_bytes=32)
        candidate = SimpleNamespace(plan=SimpleNamespace(adapter_parameters=helper.adapter_parameters))
        self.assertEqual(_phone_service_admission_reason(controller, candidate, whole, self.snapshot()),
                         "PHONE_SERVICE_MEMORY_REBALANCE_PENDING")

    def test_runtime_reserve_also_applies_to_the_smaller_declared_pool(self):
        for capped in (False, True):
            with self.subTest(capped=capped):
                controller, whole, helper = self.controller(peak=192)
                if not capped:
                    controller._phone_htp_memory_caps.clear()
                snapshot = self.snapshot(800)
                snapshot.memory.capacities["phone-memory"] = DeviceMemoryCapacity(
                    "phone-memory", 2000, 800, 64,
                )
                candidate = SimpleNamespace(plan=SimpleNamespace(adapter_parameters=helper.adapter_parameters))
                self.assertEqual(_phone_service_admission_reason(controller, candidate, whole, snapshot),
                                 "PHONE_SERVICE_MEMORY_REBALANCE_PENDING")

    def test_peak_memory_is_in_the_existing_execution_ledger_demands(self):
        _, _, helper = self.controller(peak=400)
        demands = tuple(RuntimeMemoryDemand(
            demand_id=kind, resource_id="phone-memory", kind=kind,
            required_bytes=size, resident_bytes=0, lifetime=lifetime, device_id="phone-a",
        ) for kind, size, lifetime in (("model_weights", 128, "resident"),
                                      ("kv_cache", 64, "request"), ("workspace", 32, "request")))
        profiled = RouteFeasibilityMixin._apply_runtime_memory_profile(helper, demands)
        self.assertEqual(sum(row.required_bytes for row in profiled), 400)
        self.assertEqual(profiled[:2], demands[:2])

    def test_cap_update_is_idempotent_and_transactional(self):
        controller = UnifiedScheduler.for_runtime_discovery("enforce")
        fixture, _, _ = self.controller()
        controller._runtime_capabilities = fixture._runtime_capabilities
        controller.set_phone_htp_memory_cap("phone-a", 700, workspace_bytes=32, observed_at_us=1)
        events = controller.phone_residency_events()
        controller.set_phone_htp_memory_cap("phone-a", 700, workspace_bytes=32, observed_at_us=2)
        self.assertEqual(controller.phone_residency_events(), events)
        with self.assertRaisesRegex(ValueError, "injected"):
            with controller._transaction(convert=False):
                controller.set_phone_htp_memory_cap("phone-a", 600, workspace_bytes=32, observed_at_us=3)
                raise ValueError("injected")
        self.assertEqual(controller._phone_htp_memory_caps["phone-a"], (700, 32))
        self.assertEqual(controller.phone_residency_events(), events)


class WholePhoneMemoryReuseTests(unittest.TestCase):
    peak = 10_000_000

    def setUp(self):
        self.model = manifest(6)
        self.catalog = materialize_whole_model_endpoint(
            catalog(), self.model, executor_id="executor:helper-c",
            endpoint="http://127.0.0.1:18382", backend="android-llama-server-opencl",
            adapter_parameters={
                "batch_size": 32, "context_size": 128, "cpu_device_id": "host-a",
                "executable_device": "GPUOpenCL", "execution_adapter": "android-llama-server-v1",
                "forward_port": 18382, "gpu_device_id": "helper-c", "model_alias": self.model.model_id,
                "parallel": 1, "persistent_residency": 1, "remote_library_directory": "/data/local/tmp/lib",
                "remote_model_path": "/data/local/tmp/model.gguf", "remote_port": 18382,
                "remote_server_path": "/data/local/tmp/llama-server",
                "remote_server_sha256": "sha256:" + "2" * 64, "request_io_protocol": "token-ids-v1",
                "token_id_bytes": 4, "ubatch_size": 16, "whole_model_power_prior_mw": 5000,
                "whole_model_peak_memory_bytes": self.peak,
            },
            evidence_ids=("sha256:" + "1" * 64,), transition_latency_us=1000,
            transition_energy_uj=5000, transition_energy_maturity="QUALIFIED",
            request_transport_identity_sha256="sha256:" + "3" * 64,
        )
        self.scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        self.scheduler.register_runtime_capabilities(self.catalog)
        self.scheduler.register_model_manifest(self.model)
        snapshot = runtime_snapshot(self.model)
        links = dict(snapshot.links)
        for link in self.catalog.placement_profile.links:
            if link.transport_generation == "adb-token-http-v1":
                links[link.link_id] = RuntimeLinkState(
                    link.link_id, True, link.bandwidth_bytes_per_s, 0,
                )
        self.snapshot = replace(
            snapshot, links=links,
            memory=replace(snapshot.memory, capacities={
                **snapshot.memory.capacities,
                "phone-memory": DeviceMemoryCapacity("phone-memory", self.peak + 8192, self.peak, 0),
            }),
            residency=tuple(
                replace(row, executor_id="executor:helper-c", reclaimable_bytes=self.peak,
                        resident_adapter_parameters=self.catalog.executor_by_id["executor:helper-c"].adapter_parameters)
                if row.device_id == "helper-c" else row for row in snapshot.residency
            ),
        )

    def phone_observation(self, **changes):
        return replace(self.snapshot, residency=tuple(
            replace(row, **changes) if row.device_id == "helper-c" else row
            for row in self.snapshot.residency
        ))

    def whole_candidate(self, snapshot=None, *, variant="hot"):
        snapshot = self.snapshot if snapshot is None else snapshot
        job = request("whole-memory-request")
        candidates = self.scheduler.generate_automated_candidates(
            job, self.model.model_id, snapshot,
            observed_at_us=max(job.arrival_us, snapshot.captured_at_us),
        )
        return next(row for row in candidates.candidates
                    if row.route_family == "whole_model" and row.device_ids == ("helper-c",)
                    and (variant is None or row.residency_variant == variant))

    def test_hot_observed_peak_is_not_allocated_twice(self):
        candidate = self.whole_candidate()
        self.assertTrue(candidate.admitted, candidate.rejection_reasons)
        self.assertEqual(sum(row.required_bytes for row in candidate.plan.memory_demands), self.peak)
        self.assertEqual(sum(row.additional_bytes for row in candidate.plan.memory_demands), 0)
        runtime = [row for row in candidate.plan.memory_demands if row.kind != "model_weights"]
        self.assertTrue(runtime)
        self.assertTrue(all(row.lifetime == "resident" and row.share_key for row in runtime))

    def test_real_snapshot_plan_metadata_preserves_launch_identity(self):
        plan = self.whole_candidate().plan.to_json()
        self.assertIn("request_transport_allocator", plan["adapter_parameters"])
        sample = live_executor_residency_sample(
            self.catalog, self.model, "executor:helper-c", generation=1,
            operator_plan=plan, reclaimable_bytes_by_device={"helper-c": self.peak},
        )
        snapshot = UnifiedRuntimeSnapshotBuilder(self.catalog).build(
            snapshot_id="physical-plan-memory", captured_at_us=0, valid_until_us=10_000_000,
            memory=self.snapshot.memory, residencies=(sample,),
            executor_samples={row.executor_id: EndpointRuntimeSample("healthy", "live", 1)
                              for row in self.catalog.executors},
        )
        candidate = self.whole_candidate(snapshot)
        self.assertTrue(candidate.admitted, candidate.rejection_reasons)
        self.assertEqual(sum(row.additional_bytes for row in candidate.plan.memory_demands), 0)

    def test_assumed_power_snapshot_round_trip_reuses_measured_resident_memory(self):
        profile = RuntimePhonePowerProfile.assumed_4p5w(
            device_id="helper-c", domain_id="energy:helper-c",
        )
        self.catalog = replace(self.catalog, phone_power_profiles=(profile,))
        self.scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        self.scheduler.register_runtime_capabilities(self.catalog)
        self.scheduler.register_model_manifest(self.model)
        initial = self.whole_candidate()
        self.assertTrue(initial.admitted, initial.rejection_reasons)
        plan = initial.plan.to_json()
        annotations = {key: value for key, value in plan["adapter_parameters"].items()
                       if key.startswith("phone_power_")}
        self.assertEqual(len(annotations), 6)
        self.assertEqual(annotations["phone_power_active_mw"], 4500)
        sample = live_executor_residency_sample(
            self.catalog, self.model, "executor:helper-c", generation=1,
            operator_plan=plan, reclaimable_bytes_by_device={"helper-c": self.peak},
        )
        snapshot = UnifiedRuntimeSnapshotBuilder(self.catalog).build(
            snapshot_id="physical-plan-assumed-power", captured_at_us=0, valid_until_us=10_000_000,
            memory=self.snapshot.memory, residencies=(sample,),
            executor_samples={row.executor_id: EndpointRuntimeSample("healthy", "live", 1)
                              for row in self.catalog.executors},
        )
        observed, = snapshot.residency
        for key, value in annotations.items():
            self.assertEqual(observed.resident_adapter_parameters[key], value)
        candidate = self.whole_candidate(snapshot)
        self.assertEqual(sum(row.additional_bytes for row in candidate.plan.memory_demands), 0)
        self.assertTrue(candidate.admitted, candidate.rejection_reasons)
        for key, value in annotations.items():
            self.assertEqual(candidate.plan.adapter_parameters[key], value)
        self.assertEqual(candidate.cost, initial.cost)
        compiler = self.scheduler._automated_compiler()
        self.assertEqual(
            compiler.route_template_live_state_sha256(self.snapshot),
            compiler.route_template_live_state_sha256(self.phone_observation(
                resident_adapter_parameters=plan["adapter_parameters"],
            )),
        )

    def test_launch_normalization_keeps_memory_and_execution_parameters_strict(self):
        parameters = self.catalog.executor_by_id["executor:helper-c"].adapter_parameters
        for changes in ({"whole_model_peak_memory_bytes": self.peak + 1}, {"context_size": 256},
                        {"parallel": 2}, {"batch_size": 64}, {"executable_device": "CPU"},
                        {"remote_server_sha256": "sha256:" + "9" * 64},
                        {"request_transport_generation": "other"}, {"new_launch_option": 1},
                        {"phone_power_new_launch_option": 1}):
            with self.subTest(changes=changes):
                candidate = self.whole_candidate(self.phone_observation(
                    resident_adapter_parameters={**parameters, **changes},
                ))
                self.assertFalse(candidate.admitted)
                self.assertIn("MEMORY_CAPACITY", candidate.rejection_reasons)

    def test_only_observed_bytes_receive_credit_on_cache_reuse(self):
        for missing in (4096, 20_000, 4096):
            with self.subTest(missing=missing):
                candidate = self.whole_candidate(self.phone_observation(reclaimable_bytes=self.peak - missing))
                self.assertEqual(sum(row.additional_bytes for row in candidate.plan.memory_demands), missing)
                self.assertEqual(candidate.admitted, missing <= 8192, candidate.rejection_reasons)

    def test_unknown_or_incompatible_residency_receives_no_runtime_credit(self):
        for changes in ({"reclaimable_bytes": None}, {"executor_id": None, "reclaimable_bytes": None},
                        {"executor_id": "different-service"}, {"generation": 0},
                        {"artifact_sha256": "sha256:" + "f" * 64}, {"state": "warm"},
                        {"resident_adapter_parameters": {"context_size": 256}},
                        {"resident_adapter_parameters": {"parallel": 2}}):
            with self.subTest(changes=changes):
                candidate = self.whole_candidate(self.phone_observation(**changes), variant=None)
                self.assertFalse(candidate.admitted)
                self.assertIn("MEMORY_CAPACITY", candidate.rejection_reasons)
                self.assertTrue(all(row.resident_bytes == 0 for row in candidate.plan.memory_demands
                                    if row.kind != "model_weights"))

    def test_cold_service_reserves_the_complete_peak(self):
        snapshot = replace(self.snapshot, residency=tuple(
            row for row in self.snapshot.residency if row.device_id != "helper-c"
        ))
        candidate = self.whole_candidate(snapshot, variant=None)
        self.assertFalse(candidate.admitted)
        self.assertIn("MEMORY_CAPACITY", candidate.rejection_reasons)
        self.assertEqual(sum(row.additional_bytes for row in candidate.plan.memory_demands), self.peak)

    def test_observed_launch_change_invalidates_exact_template_reuse(self):
        compiler = self.scheduler._automated_compiler()
        first = compiler.route_template_live_state_sha256(self.snapshot)
        changed = compiler.route_template_live_state_sha256(
            self.phone_observation(resident_adapter_parameters={"context_size": 256}),
        )
        self.assertNotEqual(first, changed)
        self.assertEqual(first, compiler.route_template_live_state_sha256(self.snapshot))

    def test_generated_transport_metadata_does_not_change_launch_cache_identity(self):
        compiler = self.scheduler._automated_compiler()
        plan = self.whole_candidate().plan.to_json()
        self.assertNotEqual(plan["adapter_parameters"],
                            self.catalog.executor_by_id["executor:helper-c"].adapter_parameters)
        self.assertEqual(
            compiler.route_template_live_state_sha256(self.snapshot),
            compiler.route_template_live_state_sha256(self.phone_observation(
                resident_adapter_parameters=plan["adapter_parameters"],
            )),
        )

    def test_runtime_growth_above_peak_is_a_new_request_allocation(self):
        compiler = self.scheduler._automated_compiler()
        coordinator = self.catalog.executor_by_id["executor:helper-c"]
        pattern = next(row for row in compiler._patterns(self.model)
                       if row.route_family == "whole_model" and row.device_ids == ("helper-c",))
        demands = (
            RuntimeMemoryDemand("weights:helper-c", "phone-memory", "model_weights",
                                self.model.tensor_bytes, self.model.tensor_bytes, "resident",
                                device_id="helper-c"),
            RuntimeMemoryDemand("kv:helper-c", "phone-memory", "kv_cache", 64, 0,
                                "request", device_id="helper-c"),
            RuntimeMemoryDemand("workspace:helper-c", "phone-memory", "workspace",
                                self.peak - self.model.tensor_bytes - 64 + 20_000, 0,
                                "request", device_id="helper-c"),
        )
        adjusted = compiler._persistent_whole_phone_memory_demands(
            self.model, pattern, self.snapshot, coordinator, {"helper-c": "hot"}, demands,
        )
        self.assertEqual(sum(row.required_bytes for row in adjusted), self.peak + 20_000)
        growth, = (row for row in adjusted if row.lifetime == "request")
        self.assertEqual(growth.additional_bytes, 20_000)
        self.assertIsNone(growth.share_key)
        self.assertEqual(sum(row.resident_bytes for row in adjusted), self.peak)
