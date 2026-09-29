"""Remote owners reuse authoritative phone residency without new loads."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import time

from research_dev.scheduler._internal.lifecycle import UnifiedScheduleError
from research_dev.scheduler._internal.runtime_plan import RuntimeExecutionContract, RuntimePlanError
from research_dev.scheduler._unified.automated_requests_ops.commit import _selected_phone_layout_generation
from research_dev.scheduler.adapters.phone_session_ops.identity import supports

try:
    from .test_remote_resident_launch import MANIFEST, group, command, parameters
except ImportError:
    from test_remote_resident_launch import MANIFEST, group, command, parameters


class RemoteOwnerReuseTests(unittest.TestCase):
    def fixture(self):
        contract = RuntimeExecutionContract.desktop(group())
        execution = command(contract, parameters())
        shards = contract.resident_phone_shards(MANIFEST.feed_forward_length)
        owner, = contract.remote_resident_ffn.sessions
        transport = Mock()
        transport.shares_resident_session_with.return_value = True
        phone = SimpleNamespace(
            _remote_root="/data/local/tmp/owned-test-session",
            _launch=SimpleNamespace(phone_shards=shards, transport=transport,
                remote_hashes={"model:" + MANIFEST.artifact_sha256: MANIFEST.artifact_sha256},
                weight_sources=(SimpleNamespace(session_id=owner.session_id, weight_source="ffn_shard",
                    source_sha256=owner.shard_sha256, source_path=owner.remote_path,
                    index_sha256=contract.remote_resident_ffn.shard_index_sha256,
                    parent_artifact_sha256=MANIFEST.artifact_sha256,
                    session_generation=owner.session_generation),)),
            _max_tokens_by_session={shards[0].session_id: 512},
        )
        return execution, phone, transport

    def test_verified_owner_is_reused_with_prefill_capacity(self):
        execution, phone, transport = self.fixture()
        self.assertTrue(supports(phone, execution, MANIFEST, transport))
        phone._max_tokens_by_session[phone._launch.phone_shards[0].session_id] = 1
        self.assertFalse(supports(phone, execution, MANIFEST, transport))

    def test_each_physical_identity_mismatch_refuses_reuse(self):
        for field, value in (
            ("session_generation", 6), ("layer_mask", 0xFE),
            ("maximum_columns", MANIFEST.feed_forward_length // 2),
            ("resident_bytes", 1), ("resident_geometry_sha256", "sha256:" + "8" * 64),
            ("operator_plan_sha256", "sha256:" + "8" * 64),
            ("artifact_sha256", "sha256:" + "8" * 64), ("endpoint", "session://other"),
        ):
            with self.subTest(field=field):
                execution, phone, transport = self.fixture()
                shard, = phone._launch.phone_shards
                phone._launch.phone_shards = (replace(shard, **{field: value}),)
                self.assertFalse(supports(phone, execution, MANIFEST, transport))

    def test_no_generation_zero_or_missing_operator_plan_is_materialized(self):
        for remote in (group(0), replace(group(), sessions=(replace(
                group().sessions[0], operator_plan_sha256=None),))):
            with self.subTest(remote=remote), self.assertRaises(RuntimePlanError):
                RuntimeExecutionContract.desktop(remote).resident_phone_shards(MANIFEST.feed_forward_length)

    def test_shard_and_index_hashes_must_match_the_actual_loaded_file(self):
        for field in ("source_sha256", "index_sha256", "parent_artifact_sha256", "source_path"):
            with self.subTest(field=field):
                execution, phone, transport = self.fixture()
                setattr(phone._launch.weight_sources[0], field, "different")
                self.assertFalse(supports(phone, execution, MANIFEST, transport))

    def authority(self):
        execution, phone, _ = self.fixture()
        shard, = phone._launch.phone_shards
        state = SimpleNamespace(
            state="READY", resident_artifact_sha256=shard.artifact_sha256,
            shard_geometry_sha256=shard.resident_geometry_sha256,
            operator_plan_sha256=shard.operator_plan_sha256,
            session_generation=shard.session_generation,
            resident_bytes=shard.resident_bytes, endpoint=shard.endpoint,
        )
        ready = SimpleNamespace(generation=7, layout=SimpleNamespace(shards=(shard,)))
        placement = SimpleNamespace(ready_phone_layout=lambda: ready,
            phone_session_state=lambda _: state,
            planning_phone_layout=Mock(side_effect=AssertionError("unrelated proposal is not authority")))
        selected = SimpleNamespace(plan=SimpleNamespace(
            execution_contract=execution.execution_contract, adapter_parameters=parameters()))
        return SimpleNamespace(_model_placement_controller=placement), selected, state

    def test_global_proposal_does_not_replace_per_session_ready_authority(self):
        controller, selected, _ = self.authority()
        self.assertEqual(_selected_phone_layout_generation(controller, selected, "calibration", None, None), 7)

    def test_session_loss_or_generation_change_fails_commit(self):
        for field, value in (("state", "LOADING"), ("session_generation", 6),
                             ("resident_artifact_sha256", "sha256:" + "8" * 64)):
            with self.subTest(field=field):
                controller, selected, state = self.authority()
                setattr(state, field, value)
                with self.assertRaisesRegex(UnifiedScheduleError, "READY authority"):
                    _selected_phone_layout_generation(controller, selected, "calibration", None, None)


class RemoteOwnerAdmissionTests(unittest.TestCase):
    def test_published_owner_generates_a_bound_candidate_and_real_phone_leases(self):
        self._published_owner()

    def test_new_context_keeps_qualified_preload_separate_from_execution(self):
        self._published_owner(new_context=True)

    def _published_owner(self, *, new_context=False):
        try:
            from .test_offline_phone_residency import OfflinePhoneResidencyTests, _OfflinePhoneBackend
        except ImportError:
            from test_offline_phone_residency import OfflinePhoneResidencyTests, _OfflinePhoneBackend
        from research_dev.scheduler.campaigns.burstgpt.remote_resident_gate import (
            _phone_preload_catalog, _remote_owner_catalog,
        )
        from research_dev.scheduler.adapters import CanonicalOfflinePhoneResidencyPreloader, interpret_runtime_ticket
        OfflinePhoneResidencyTests.setUpClass()
        scheduler, requests, snapshot = OfflinePhoneResidencyTests.replay_runtime(load_evidence=False)
        catalog = scheduler._runtime_capabilities
        # This synthetic transport supports resident prefill, unlike the decode-only replay links.
        fixture_identity = "sha256:" + "d" * 64
        catalog = replace(catalog, placement_profile=replace(catalog.placement_profile, links=tuple(
            replace(row, maximum_payload_bytes=16 << 20, usbfs_available_bytes=256 << 20,
                    qualification_identity_sha256=fixture_identity, evidence_ids=(fixture_identity,))
            if row.transport_generation.startswith("functionfs") else row
            for row in catalog.placement_profile.links)))
        manifest = scheduler.runtime_model_manifest(next(iter(requests)))
        source = catalog.composite_executor_by_id[
            catalog.desktop_control_by_artifact[manifest.artifact_sha256].executor_id]
        if new_context:
            prior = catalog
            context = source.adapter_parameters["context_resource_id"]
            catalog = replace(catalog, resources={**catalog.resources, context: replace(
                catalog.resources[context], capacity=catalog.resources[context].capacity * 2)},
                composite_executors=tuple(replace(row,
                    maturity="QUALIFIED" if row.executor_id == source.executor_id else "SHADOW",
                    adapter_parameters={**row.adapter_parameters,
                        "context_size": source.adapter_parameters["context_size"] * 2})
                    if row.executor_id == source.executor_id or row.baseline_executor_id == source.executor_id
                    else row for row in catalog.composite_executors))
            source = catalog.composite_executor_by_id[source.executor_id]
            registered = tuple(scheduler._runtime_manifests.values())
            scheduler = type(scheduler).for_runtime_discovery(
                "enforce", maximum_phone_sessions=scheduler._maximum_phone_sessions)
            scheduler.register_runtime_capabilities(catalog)
            for registered_manifest in registered:
                scheduler.register_model_manifest(registered_manifest)
            preload = _phone_preload_catalog(catalog, prior, manifest.artifact_sha256)
            changed = replace(source, adapter_parameters={**source.adapter_parameters, "batch_size": 1})
            invalid = replace(catalog, composite_executors=tuple(
                changed if row.executor_id == source.executor_id else row for row in catalog.composite_executors))
            with self.assertRaisesRegex(RuntimeError, "launch differs beyond context"):
                _phone_preload_catalog(invalid, prior, manifest.artifact_sha256)
            with self.assertRaisesRegex(RuntimeError, "devices or transport differ"):
                _phone_preload_catalog(replace(catalog, placement_profile=replace(
                    catalog.placement_profile, links=())), prior, manifest.artifact_sha256)
            scheduler.register_runtime_capabilities(preload)
        else:
            scheduler.register_runtime_capabilities(catalog)
        plan = scheduler.plan_offline_phone_residency(requests, snapshot=snapshot,
                                                     observed_at_us=snapshot.captured_at_us)
        index = SimpleNamespace(index_sha256="sha256:" + "a" * 64,
            resolve=lambda *args, **kwargs: SimpleNamespace(
                shard_sha256="sha256:" + "b" * 64, remote_path="/data/local/tmp/ffn.gguf"))
        catalog, remote = _remote_owner_catalog(catalog, manifest, source, plan.target_layout, index)
        if not new_context:
            scheduler.register_runtime_capabilities(catalog)
        snapshot = replace(snapshot, executors={**snapshot.executors,
            remote.executor_id: replace(snapshot.executors[source.executor_id], executor_id=remote.executor_id)})
        backend = _OfflinePhoneBackend(snapshot)
        preloader = CanonicalOfflinePhoneResidencyPreloader(scheduler, backend,
            epoch_ns=time.monotonic_ns() - snapshot.captured_at_us * 1000,
            snapshot_provider=backend.snapshot)
        prepared = preloader.preload(plan, lambda stage: object(), initial_snapshot=snapshot,
                                     observed_at_us=snapshot.captured_at_us)
        if new_context:
            ready = scheduler._model_placement_controller.ready_phone_layout()
            scheduler.register_runtime_capabilities(catalog)
            self.assertEqual(ready, scheduler._model_placement_controller.ready_phone_layout())
            self.assertEqual(remote.maturity, "CALIBRATION_PENDING")
            self.assertTrue(all(command.transition.prepares_device_ids == (remote.adapter_parameters["phone_device_id"],)
                                for command in prepared.commands))
        fresh = backend.snapshot(prepared.plan.current_stage, backend.clock_us + 1)
        request = replace(requests[manifest.model_id][0], request_id="remote-owner-request",
                          arrival_us=fresh.captured_at_us)
        candidates = scheduler.generate_automated_candidates(request, manifest.model_id, fresh,
                                                            observed_at_us=fresh.captured_at_us)
        candidate, = (row for row in candidates.candidates if row.binding.executor_id == remote.executor_id)
        self.assertTrue(candidate.plan.execution_contract.remote_resident_ffn.bound, candidate.rejection_reasons)
        self.assertFalse(any("OWNER_NOT_READY" in reason or reason == "RESIDENCY_TRANSITION_ABSENT"
                             for reason in candidate.rejection_reasons), candidate.rejection_reasons)
        owner_device = remote.adapter_parameters["phone_device_id"]
        self.assertIn(owner_device, candidate.device_ids)
        self.assertTrue(any(row.device_id == owner_device and row.kind == "workspace"
                            for row in candidate.plan.memory_demands))
        self.assertTrue(all(row.required_bytes == row.resident_bytes
                            for row in candidate.plan.memory_demands
                            if row.kind == "session_residency_constraint"))
        ticket = scheduler.submit_automated_request(request, manifest.model_id, fresh,
            observed_at_us=fresh.captured_at_us, selection_mode="calibration")
        self.assertEqual(ticket.binding.executor_id, remote.executor_id, candidate.rejection_reasons)
        command = interpret_runtime_ticket(ticket)
        self.assertTrue(command.leases)
        self.assertTrue(command.memory_reservations)
        self.assertGreater(command.phone_layout_generation, 0)
        self.assertTrue(set(remote.participant_resource_ids[owner_device]).issubset(
            row["resource_id"] for row in command.leases))
        from research_dev.scheduler.adapters.phone_transport import phone_transport_contract
        transport = phone_transport_contract(command.adapter_parameters)
        self.assertEqual(transport.transport, "functionfs-usb")
        self.assertEqual(transport.qualification_identity_sha256, fixture_identity)
        self.assertGreaterEqual(transport.max_payload_bytes,
            manifest.embedding_length * source.adapter_parameters["ubatch_size"] * 2)
        self.assertTrue(any(row["resource_id"].startswith("link:usb-") for row in command.leases))
        from research_dev.scheduler.adapters import llama_server_launch_contract
        launch = llama_server_launch_contract(command, manifest)
        self.assertEqual(launch.ffn_environment["S41_SERVER_FFN_TRANSPORT"], "functionfs-usb")
        self.assertEqual(int(launch.ffn_environment["S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK"]),
                         command.execution_contract.remote_resident_ffn.layer_mask)
        self.assertEqual(launch.ffn_environment["S41_SERVER_FFN_RUNTIME_CONTROL"], "1")


class RemoteOwnerTransportTests(unittest.TestCase):
    def test_only_ready_measured_links_to_the_exact_owner_are_used(self):
        from research_dev.scheduler._internal.route_generation.remote_resident import remote_resident_link_ids
        base = dict(source_device="cpu", target_device="phone", ready=True, status="measured",
                    transport_generation="functionfs-dmabuf-async-ring-v2")
        links = [SimpleNamespace(link_id="valid", **base)]
        for key, override in (("not-ready", {"ready": False}),
                              ("unknown", {"status": "estimated"}),
                              ("other-phone", {"target_device": "other"}),
                              ("token-rpc", {"transport_generation": "adb-token-http-v1"})):
            links.append(SimpleNamespace(link_id=key, **{**base, **override}))
        profile = SimpleNamespace(links=links)
        params = {"remote_resident_ffn_v1": "declaration", "cpu_device_id": "cpu", "phone_device_id": "phone"}
        self.assertEqual(remote_resident_link_ids(profile, params), ("valid",))
        self.assertEqual(remote_resident_link_ids(profile, {}), ())

    def test_missing_transport_cannot_fall_back_to_an_implicit_bridge(self):
        from research_dev.scheduler._internal.route_generation.costing_parameters import RouteParameterMixin
        compiler = SimpleNamespace(_transport_adapter_parameters=Mock(return_value={}))
        reason = RouteParameterMixin._candidate_transport_parameters(
            compiler, None, (), None, SimpleNamespace(assistance_phase="decode"),
            {"remote_resident_ffn_v1": "declaration"})
        self.assertEqual(reason, "remote-resident phone transport is not qualified")


if __name__ == "__main__":
    unittest.main()
