"""Remote-resident owner trace arm (Phase 2, target A): opt-in configuration, the scheduler pin,
the runner's owner-readiness wait, the transport contract, preflight and the RESULT ledger.

The arm's claim is that the desktop never loads the owner layers, so every path that could reach
the full-weight desktop parent of the pinned model (selection, placement epochs, startup preload,
recovery) must fail closed instead.
"""
from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field, replace
import json
import os
import sys
import time
import unittest
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import UnifiedScheduleError  # noqa: E402
from research_dev.scheduler.adapters import (  # noqa: E402
    CanonicalOfflinePhoneResidencyPreloader,
    PhysicalAdapterError,
    interpret_runtime_ticket,
    llama_server_launch_contract,
    materialize_remote_resident_owner,
    phone_transport_contract,
    remote_resident_owner_executor_id,
)
from research_dev.scheduler.adapters.phone_helpers import (  # noqa: E402
    PhoneHelperBinding,
    phone_helper_launch_environment,
)
from research_dev.scheduler.adapters.preflight import (  # noqa: E402
    _check_remote_resident_owner,
    remote_resident_route_blockers,
)
from research_dev.scheduler.config import (  # noqa: E402
    CAMPAIGN_MANIFEST_SCHEMA,
    CampaignManifest,
    FixedPhoneResidencyConfiguration,
    RemoteResidentOwnerConfiguration,
    SchedulerConfigurationError,
    StartupDesktopParentConfiguration,
)
from research_dev.scheduler.campaigns.burstgpt import runner  # noqa: E402
from research_dev.scheduler._internal.adaptive_decode_contracts import AdaptiveDecodeError  # noqa: E402
from research_dev.scheduler._internal.adaptive_decode_planning import _policy  # noqa: E402
from research_dev.scheduler._internal.route_generation import AutomatedRouteCompiler  # noqa: E402
from research_dev.scheduler._internal.route_generation.common import RouteGenerationError  # noqa: E402
from research_dev.scheduler._internal.runtime_plan import RuntimeRemoteResidentFfn  # noqa: E402
from research_dev.scheduler._unified.adaptive_decode_control import AdaptiveDecodeControlMixin  # noqa: E402
from research_dev.scheduler._internal.runtime_controller_ops.completion import _route_violation  # noqa: E402
from research_dev.scheduler._unified.automated_selection_ops.calibration import (  # noqa: E402
    _select_remote_resident_owner_candidate,
)
from research_dev.scheduler._unified.automated_requests_ops.failure import (  # noqa: E402
    _prepare_automated_failure_fallback,
)
from research_dev.scheduler.adapters.heterogeneous_rig_ops.transitions import RigTransitionMixin  # noqa: E402

try:
    from . import test_offline_phone_residency as offline_fixture
    from . import test_remote_resident_materialization as owner
    from . import test_remote_resident_routes as route_fixture
    from . import two_phone_harness as two_phone
except ImportError:
    import test_offline_phone_residency as offline_fixture
    import test_remote_resident_materialization as owner
    import test_remote_resident_routes as route_fixture
    import two_phone_harness as two_phone


# the production arm's shape: three owner sessions, one per HTP worker (GGML_HEXAGON_NDEV=3)
THREE_OWNERS = {"HTP0": 0x3F, "HTP1": 0x3F << 6, "HTP2": 0x3F << 12}


def pinned_arm(*, pin: bool = True, masks=THREE_OWNERS):
    """The replay rig with a materialized remote-resident parent and its fixed owner sessions."""
    fixture = owner.owner_fixture()
    manifest = fixture.manifest
    catalog = materialize_remote_resident_owner(
        fixture.catalog, manifest, masks, owner.shard_index(manifest, masks))
    control = catalog.desktop_control_by_artifact[manifest.artifact_sha256]
    remote_id = remote_resident_owner_executor_id(control.executor_id)
    scheduler = fixture.scheduler
    scheduler.register_runtime_capabilities(catalog)
    scheduler.configure_fixed_phone_residency(owner.fixed_residency(manifest, masks))
    if pin:
        scheduler.configure_remote_resident_owner(manifest.model_id)
    snapshot = replace(fixture.snapshot, executors={
        **fixture.snapshot.executors,
        remote_id: replace(fixture.snapshot.executors[control.executor_id], executor_id=remote_id),
    })
    return SimpleNamespace(scheduler=scheduler, manifest=manifest, catalog=catalog, control=control,
                           remote_id=remote_id, snapshot=snapshot, requests=fixture.requests[manifest.model_id])


def load_owners(arm, *, sessions: int | None = None) -> object:
    """Fixed-residency preload through the canonical preloader, one stage per owner session;
    ``sessions`` stops it after that many owners. Returns a snapshot after the last load."""
    scheduler, snapshot = arm.scheduler, arm.snapshot
    plan = scheduler.plan_offline_phone_residency(
        scheduler.fixed_phone_residency_requests(), snapshot=snapshot, observed_at_us=snapshot.captured_at_us)
    backend = offline_fixture._OfflinePhoneBackend(snapshot)
    preloader = CanonicalOfflinePhoneResidencyPreloader(
        scheduler, backend, epoch_ns=time.monotonic_ns() - snapshot.captured_at_us * 1000,
        snapshot_provider=backend.snapshot)
    stages = []

    def payload(stage):
        if len(stages) == sessions:
            raise RuntimeError("owner preload stopped")
        stages.append(stage)
        return object()

    try:
        prepared = preloader.preload(plan, payload, initial_snapshot=snapshot,
                                     observed_at_us=snapshot.captured_at_us)
        assert prepared.plan.state == "READY" and sessions is None
        stage = prepared.plan.current_stage
    except RuntimeError:
        stage = scheduler.offline_phone_residency_stage(plan.plan_id)
    return backend.snapshot(stage, backend.clock_us + 1)


def arrival(arm, request_id: str, at_us: int, index: int = 0):
    return replace(arm.requests[index % len(arm.requests)], request_id=request_id, arrival_us=at_us)


class CampaignConfigurationTests(unittest.TestCase):
    ROW = {"schema": CAMPAIGN_MANIFEST_SCHEMA, "campaign_id": "remote-owner-test",
           "rig_manifest_path": "rig.json", "models_manifest_path": "models.json",
           "evidence_manifest_path": "evidence.json", "selection_mode": "energy-aware",
           "maximum_latency_ppm": 1_100_000, "energy_attribution_kind": "diagnostic",
           "trace": {"large_requests_path": "large.json", "overlay_requests_path": "overlay.json",
                     "trace_manifest_path": "trace.json"}}
    FIXED = FixedPhoneResidencyConfiguration("q32-owner", (
        ("HTP0", "sha256:" + "a" * 64, 0x7FF, 25600), ("HTP1", "sha256:" + "a" * 64, 0x1FF800, 25600),
    )).to_json()
    OWNER = {"model_id": "qwen3-32b-q4_0", "owner_ready_timeout_s": 900}

    def test_absent_by_default_and_round_trips_when_set(self) -> None:
        plain = CampaignManifest.from_json(dict(self.ROW), Path("/inputs"))
        self.assertIsNone(plain.remote_resident_owner)
        self.assertNotIn("remote_resident_owner", plain.to_json())
        configured = CampaignManifest.from_json(
            {**self.ROW, "fixed_phone_residency": self.FIXED, "remote_resident_owner": self.OWNER}, Path("/inputs"))
        self.assertEqual(configured.remote_resident_owner, RemoteResidentOwnerConfiguration("qwen3-32b-q4_0", 900))
        self.assertEqual(CampaignManifest.from_json(configured.to_json(), Path("/inputs")), configured)
        self.assertEqual(RemoteResidentOwnerConfiguration.from_json({"model_id": "m"}).owner_ready_timeout_s, 1800)

    def test_invalid_or_unsafe_combinations_fail_closed(self) -> None:
        startup = StartupDesktopParentConfiguration("qwen3-32b-q4_0", "physical:hot:desktop", "sha256:" + "b" * 64)
        cases = (
            ({"remote_resident_owner": self.OWNER}, "fixed phone residency"),
            ({"fixed_phone_residency": self.FIXED, "remote_resident_owner": self.OWNER,
              "selection_mode": "desktop-baseline"}, "desktop-baseline"),
            ({"fixed_phone_residency": self.FIXED, "remote_resident_owner": self.OWNER,
              "include_startup_preparation": True, "startup_desktop_parents": [startup.to_json()]},
             "startup desktop parent"),
            ({"fixed_phone_residency": self.FIXED, "remote_resident_owner": {**self.OWNER, "extra": 1}}, "fields"),
            ({"fixed_phone_residency": self.FIXED, "remote_resident_owner": {"owner_ready_timeout_s": 5}}, "fields"),
            ({"fixed_phone_residency": self.FIXED,
              "remote_resident_owner": {**self.OWNER, "owner_ready_timeout_s": 0}}, "timeout"),
        )
        for changes, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(SchedulerConfigurationError, message):
                CampaignManifest.from_json({**self.ROW, **changes}, Path("/inputs"))

    def test_launch_commands_carry_the_opt_in_and_the_model_must_own_phone_shards(self) -> None:
        from test_two_phone_helpers import CampaignConfigurationTests as Fixture, legacy_rig_json
        from research_dev.scheduler.campaigns.burstgpt import launch
        from research_dev.scheduler.config import load_scheduler_configuration

        case = Fixture()
        case.setUp()
        self.addCleanup(case.tearDown)
        path = case._write(legacy_rig_json(), None)
        plain = load_scheduler_configuration(path, environ={})

        def commands(configuration):
            return (launch.preflight_command(configuration, catalog_path=case.root / "c.json",
                                             normal_usb_receipt_path=case.root / "u.json",
                                             output_path=case.root / "out.json"),)

        for command in commands(plain):
            self.assertNotIn("--remote-resident-owner-json", command)
        campaign = json.loads(path.read_text())
        campaign["fixed_phone_residency"] = self.FIXED
        campaign["remote_resident_owner"] = {**self.OWNER, "model_id": "qwen3-14b-q4km-dequant-f16"}
        path.write_text(json.dumps(campaign))
        with self.assertRaisesRegex(SchedulerConfigurationError, "phone FFN shards"):
            load_scheduler_configuration(path, environ={})
        models_path = case.root / "models.json"
        models = json.loads(models_path.read_text())
        models["models"][0].update(phone_ffn_shard_index_path="FFN_SHARDS.json",
                                   phone_ffn_shard_directory="/data/local/tmp/q32")
        models_path.write_text(json.dumps(models))
        configured = load_scheduler_configuration(path, environ={})
        for command in commands(configured):
            value = json.loads(command[command.index("--remote-resident-owner-json") + 1])
            self.assertEqual(value, {"model_id": "qwen3-14b-q4km-dequant-f16", "owner_ready_timeout_s": 900})
        source = Path(launch.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count('"--remote-resident-owner-json"'), 2)  # runner and preflight commands
        self.assertIn("--remote-resident-owner-json", runner._build_parser()._option_string_actions)


class SchedulerPinTests(unittest.TestCase):
    def test_owners_not_ready_fail_closed_without_a_full_weight_fallback(self) -> None:
        arm = pinned_arm()
        at = arm.snapshot.captured_at_us
        for mode in ("energy-aware", "adaptive-decode", "calibration"):
            with self.subTest(mode=mode), self.assertRaisesRegex(
                    UnifiedScheduleError, "not admissible.*REMOTE_RESIDENT_OWNER_NOT_READY:HTP0"):
                arm.scheduler.submit_automated_request(
                    arrival(arm, "early-" + mode, at), arm.manifest.model_id, arm.snapshot,
                    observed_at_us=at, selection_mode=mode)
        self.assertEqual(arm.scheduler._runtime_controller.current_tickets(), ())

    def test_every_owner_session_must_be_ready(self) -> None:
        arm = pinned_arm()
        partial = load_owners(arm, sessions=2)
        with self.assertRaisesRegex(UnifiedScheduleError, "not admissible") as raised:
            arm.scheduler.submit_automated_request(
                arrival(arm, "two-of-three", partial.captured_at_us), arm.manifest.model_id, partial,
                observed_at_us=partial.captured_at_us, selection_mode="energy-aware")
        self.assertIn("REMOTE_RESIDENT_OWNER_NOT_READY:HTP2", str(raised.exception))
        self.assertNotIn("NOT_READY:HTP0", str(raised.exception))
        self.assertEqual(arm.scheduler._runtime_controller.current_tickets(), ())

    def test_ready_owners_pin_the_remote_parent_in_every_campaign_mode(self) -> None:
        arm = pinned_arm()
        fresh = load_owners(arm)
        baseline_route = f"auto:coordinated:{arm.control.executor_id}:residency:cold"
        for index, mode in enumerate(("energy-aware", "adaptive-decode", "calibration")):
            at = fresh.captured_at_us + index
            ticket = arm.scheduler.submit_automated_request(
                arrival(arm, "pinned-" + mode, at, index), arm.manifest.model_id, fresh,
                observed_at_us=at, selection_mode=mode)
            with self.subTest(mode=mode):
                self.assertEqual(ticket.binding.executor_id, arm.remote_id)
                self.assertEqual(ticket.selection_mode, "calibration")
                self.assertEqual(ticket.decision.reason, "REMOTE_RESIDENT_OWNER_PINNED")
                self.assertEqual(dict(ticket.decision.rejected)[baseline_route], "REMOTE_RESIDENT_OWNER_PINNED")
                self.assertIsNotNone(ticket.execution_plan.execution_contract.remote_resident_ffn)
        launch = llama_server_launch_contract(interpret_runtime_ticket(ticket), arm.manifest)
        environment = launch.ffn_environment
        self.assertEqual(environment["S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK"], str(0x3FFFF))
        self.assertEqual(environment["S41_SERVER_FFN_SHARDS"], ";".join(
            f"{session}@session://op15-phone/{session}:{mask}" for session, mask in THREE_OWNERS.items()))
        self.assertEqual(environment["S41_SERVER_FFN_MAX_TOKENS"], "512")
        self.assertEqual(int(environment["S41_SERVER_FFN_USB_MAX_PAYLOAD_BYTES"]),
                         arm.manifest.embedding_length * 512 * 2)
        self.assertNotIn("S41_SERVER_FFN_FAST_HASH", environment)

    def test_owner_requests_share_the_phone_lanes_of_their_server(self) -> None:
        # pB1q8oc 2026-10-10 ran its owner requests one at a time: each leased the OnePlus lanes
        # (capacity one) exclusively. The owner server batches all its requests into one call per
        # layer, so they lease the phone lanes as co-tenants and overlap; desktop leases stay theirs.
        arm = pinned_arm()
        fresh = load_owners(arm)
        at = fresh.captured_at_us
        first = arm.scheduler.submit_automated_request(
            arrival(arm, "tenant-a", at), arm.manifest.model_id, fresh,
            observed_at_us=at, selection_mode="energy-aware")
        second = arm.scheduler.submit_automated_request(
            arrival(arm, "tenant-b", at + 1, 1), arm.manifest.model_id, fresh,
            observed_at_us=at + 1, selection_mode="energy-aware")
        for ticket in (first, second):
            self.assertEqual(ticket.binding.executor_id, arm.remote_id)
        self.assertLess(second.decision.start_us, first.decision.finish_us)
        phone = arm.catalog.executor_by_device[
            first.execution_plan.adapter_parameters["phone_device_id"]]
        keys = {
            (ticket.request.request_id, lease.resource_id): lease.share_key
            for ticket in (first, second) for lease in ticket.decision.leases
        }
        shared = {key for (_, resource_id), key in keys.items() if resource_id in phone.execution_resource_ids}
        self.assertEqual(len(shared), 1)
        self.assertTrue(next(iter(shared)).startswith("remote-resident-server:" + arm.remote_id + "@"))
        self.assertTrue(any(
            key is None for (_, resource_id), key in keys.items()
            if resource_id not in phone.execution_resource_ids))

    def test_a_full_owner_server_queues_an_arrival_instead_of_refusing_it(self) -> None:
        # pB1q8oa 2026-10-10: with four co-tenants on the owner server (no free slot) a replan was
        # refused EXECUTOR_CAPACITY_UNAVAILABLE and failed the run: the phone's owner sessions are
        # held under the preloading executor, so the scheduler-owned wait never recognized them
        arm = pinned_arm()
        fresh = load_owners(arm)
        at = fresh.captured_at_us
        for index in range(4):
            arm.scheduler.submit_automated_request(
                arrival(arm, f"tenant-{index}", at + index, index), arm.manifest.model_id, fresh,
                observed_at_us=at + index, selection_mode="energy-aware")
        full = replace(fresh, executors={
            **fresh.executors,
            arm.remote_id: replace(fresh.executors[arm.remote_id], free_slots=0),
        })
        fifth = arm.scheduler.submit_automated_request(
            arrival(arm, "tenant-4", at + 4, 4), arm.manifest.model_id, full,
            observed_at_us=at + 4, selection_mode="energy-aware")
        self.assertEqual(fifth.binding.executor_id, arm.remote_id)

    def test_without_the_pin_the_same_catalog_keeps_the_desktop_baseline(self) -> None:
        arm = pinned_arm(pin=False)
        fresh = load_owners(arm)
        ticket = arm.scheduler.submit_automated_request(
            arrival(arm, "unpinned", fresh.captured_at_us), arm.manifest.model_id, fresh,
            observed_at_us=fresh.captured_at_us, selection_mode="energy-aware")
        self.assertEqual(ticket.binding.executor_id, arm.control.executor_id)
        self.assertEqual(ticket.selection_mode, "energy-aware")

    def test_full_weight_startup_preload_is_refused(self) -> None:
        arm = pinned_arm()
        fresh = load_owners(arm)
        request = replace(arrival(arm, "startup", fresh.captured_at_us), input_tokens=1, output_tokens=2,
                          quality_requirement="exact")
        with self.assertRaisesRegex(UnifiedScheduleError, "forbids the full-weight route"):
            arm.scheduler.submit_startup_parent_preload(
                request, arm.manifest.model_id, fresh, executor_id=arm.control.executor_id,
                desktop_placement_sha256=arm.control.placement_sha256, observed_at_us=fresh.captured_at_us)

    def test_pin_configuration_fails_closed(self) -> None:
        fixture = owner.owner_fixture()
        manifest = fixture.manifest
        catalog = materialize_remote_resident_owner(
            fixture.catalog, manifest, owner.OWNER_MASKS, owner.shard_index(manifest))
        scheduler = fixture.scheduler
        scheduler.register_runtime_capabilities(fixture.catalog)
        scheduler.configure_fixed_phone_residency(owner.fixed_residency(manifest))
        with self.assertRaisesRegex(UnifiedScheduleError, "exactly one remote-resident parent"):
            scheduler.configure_remote_resident_owner(manifest.model_id)
        scheduler.register_runtime_capabilities(catalog)
        with self.assertRaisesRegex(UnifiedScheduleError, "not registered"):
            scheduler.configure_remote_resident_owner("another-model")
        scheduler.configure_remote_resident_owner(manifest.model_id)
        self.assertEqual(scheduler.remote_resident_owner_configuration()["executor_id"],
                         remote_resident_owner_executor_id(catalog.desktop_control_by_artifact[
                             manifest.artifact_sha256].executor_id))
        with self.assertRaisesRegex(UnifiedScheduleError, "precede every request"):
            scheduler.configure_remote_resident_owner(manifest.model_id)
        other = owner.owner_fixture().scheduler
        other.register_runtime_capabilities(catalog)
        other.configure_fixed_phone_residency(owner.fixed_residency(
            manifest, {**owner.OWNER_MASKS, "HTP2": 0x3F << 12}))
        with self.assertRaisesRegex(UnifiedScheduleError, "differ from the fixed residency"):
            other.configure_remote_resident_owner(manifest.model_id)

    def test_pinned_parent_admits_an_uncalibrated_join_but_not_a_physical_blocker(self) -> None:
        # pB1q8oc 2026-10-10: an arrival joining running owner work carried MARGINAL_SYSTEM_COST_UNKNOWN;
        # the pinned parent has no other route, so the cost-evidence reason must not refuse it
        @dataclass(frozen=True)
        class Binding:
            ready: bool = False
            eligibility_reasons: tuple = ("MARGINAL_SYSTEM_COST_UNKNOWN",)

        @dataclass(frozen=True)
        class Row:
            candidate_id: str
            rejection_reasons: tuple
            binding: Binding = Binding()
            admitted: bool = False
            plan: object = field(default_factory=lambda: SimpleNamespace(
                execution_contract=SimpleNamespace(remote_resident_ffn=object())))
            cost: object = field(default_factory=lambda: SimpleNamespace(finish_upper_us=1, finish_us=1))

        controller = SimpleNamespace(
            _candidate_quarantine_reason=lambda row: None,
            _calibration_physically_qualified=lambda row: True,
            _runtime_decode_cohorts=SimpleNamespace(can_join=lambda *args, **kwargs: False))

        def select(*rows):
            return _select_remote_resident_owner_candidate(controller, SimpleNamespace(
                candidate_set=SimpleNamespace(model_id="pinned", candidates=rows), excluded=(),
                runtime_rejection={}, request=SimpleNamespace(quality_requirement="semantic")))

        selected, rejected, reason = select(Row("owner", ("MARGINAL_SYSTEM_COST_UNKNOWN", "ROUTE_NOT_QUALIFIED")))
        self.assertEqual((selected.candidate_id, selected.admitted, selected.binding.ready), ("owner", True, True))
        self.assertEqual((rejected, reason), ((), "REMOTE_RESIDENT_OWNER_PINNED"))
        with self.assertRaisesRegex(UnifiedScheduleError, "owner=MEMORY_CAPACITY"):
            select(Row("owner", ("MARGINAL_SYSTEM_COST_UNKNOWN", "MEMORY_CAPACITY")))

    def test_a_pinned_latency_miss_asks_for_recalibration_instead_of_quarantine(self) -> None:
        # pB1q8oc 2026-10-10: owner requests ended past their ASSUMED-cost upper bounds; quarantining the
        # model's only route failed the next arrival
        def violation(reason, kind):
            controller = SimpleNamespace(_route_uncertainty={}, _quarantined_routes=set())
            ticket = SimpleNamespace(
                cost_estimates=SimpleNamespace(estimates=(SimpleNamespace(route_id="owner", baseline=False),)),
                decision=SimpleNamespace(route_id="owner", reason=reason, finish_upper_us=5))
            action = _route_violation(controller, ticket, 10, kind)
            return action, controller._quarantined_routes, controller._route_uncertainty["owner"]

        action, quarantined, state = violation("REMOTE_RESIDENT_OWNER_PINNED", "prediction")
        self.assertEqual((action, quarantined), ("pinned_route_requires_recalibration", set()))
        self.assertEqual(state["upper_bound_violations"], 1)
        self.assertEqual(violation("REMOTE_RESIDENT_OWNER_PINNED", "physical")[:2],
                         ("disabled_for_remaining_run", {"owner"}))
        self.assertEqual(violation("PHYSICAL_CALIBRATION_SAMPLE", "prediction")[:2],
                         ("disabled_for_remaining_run", {"owner"}))

    def test_pinned_failures_have_no_recovery_route(self) -> None:
        controller = SimpleNamespace(_remote_resident_owner=MappingProxyType({"model_id": "pinned"}))
        with self.assertRaisesRegex(UnifiedScheduleError, "no recovery route"):
            _prepare_automated_failure_fallback(
                controller, current=None, manifest=SimpleNamespace(model_id="pinned"), snapshot=None,
                failed=None, failed_at_us=0)

    def test_untransportable_parent_is_refused_with_its_reasons(self) -> None:
        arm = pinned_arm()
        fresh = load_owners(arm)
        # the identity qualifies one prefill row per message only (split-row), not the whole ubatch
        small = replace(arm.catalog, placement_profile=replace(arm.catalog.placement_profile, links=tuple(
            replace(row, maximum_payload_bytes=arm.manifest.embedding_length * 2 * 4)
            if row.transport_generation.startswith("functionfs") else row
            for row in arm.catalog.placement_profile.links)))
        arm.scheduler.register_runtime_capabilities(small)
        with self.assertRaisesRegex(UnifiedScheduleError, "TRANSPORT_PROFILE_INCOMPLETE"):
            arm.scheduler.submit_automated_request(
                arrival(arm, "small-payload", fresh.captured_at_us), arm.manifest.model_id, fresh,
                observed_at_us=fresh.captured_at_us, selection_mode="energy-aware")


class TransportTests(unittest.TestCase):
    def test_fast_hash_is_an_opt_in_of_the_functionfs_contract(self) -> None:
        parameters = dict(two_phone.FUNCTIONFS_PARAMETERS)
        plain = phone_transport_contract(parameters)
        fast = phone_transport_contract({**parameters, "ffn_fast_hash": 1})
        self.assertNotIn("S41_SERVER_FFN_FAST_HASH", plain.server_environment())
        self.assertEqual(fast.server_environment()["S41_SERVER_FFN_FAST_HASH"], "1")
        self.assertEqual(dict(phone_transport_contract({**parameters, "ffn_fast_hash": 0}).server_environment()),
                         dict(plain.server_environment()))
        self.assertFalse(plain.shares_resident_session_with(fast))
        for invalid in ({**parameters, "ffn_fast_hash": 2}, {**parameters, "ffn_fast_hash": True},
                        {**two_phone.PIXEL_TRANSPORT, "ffn_fast_hash": 1}):
            with self.subTest(invalid=invalid), self.assertRaises(PhysicalAdapterError):
                phone_transport_contract(invalid)

    def test_fast_hash_follows_its_helper_in_a_several_phone_environment(self) -> None:
        primary = {**two_phone.FUNCTIONFS_PARAMETERS, "ffn_fast_hash": 1, "phone_device_id": two_phone.OP15}
        helpers = tuple(PhoneHelperBinding(**row) for row in json.loads(two_phone.phone_helpers_json(0b1111)))
        shared = {"S41_SERVER_FFN_LAYER_MASK": str(0b1111 | two_phone.PIXEL_MASK),
                  **phone_transport_contract(primary).server_environment()}
        environment = phone_helper_launch_environment(helpers, shared, primary)
        self.assertEqual(environment["S41_SERVER_FFN_HELPER0_FAST_HASH"], "1")
        self.assertNotIn("S41_SERVER_FFN_FAST_HASH", environment)
        self.assertNotIn("S41_SERVER_FFN_HELPER1_FAST_HASH", environment)

    def test_remote_resident_routes_qualify_the_whole_ubatch_payload(self) -> None:
        catalog = owner.owner_fixture(payload_bytes=10_240 * 128).catalog
        profile = catalog.placement_profile
        links = tuple(row.link_id for row in profile.links if row.transport_generation.startswith("functionfs"))
        needed = 5120 * 512 * 2
        split = AutomatedRouteCompiler._transport_adapter_parameters(
            profile, links, needed, 1, None, "split-row", 512)
        self.assertEqual((split["usb_max_payload_bytes"], split["usb_qualified_transfer_payload_bytes"]),
                         (needed, needed // 512))
        with self.assertRaisesRegex(RouteGenerationError, "maximum payload is not qualified"):
            AutomatedRouteCompiler._transport_adapter_parameters(
                profile, links, needed, 1, None, "split-row", 512, whole_payload=True)
        whole = owner.owner_fixture(payload_bytes=needed).catalog.placement_profile
        value = AutomatedRouteCompiler._transport_adapter_parameters(
            whole, links, needed, 1, None, "split-row", 512, whole_payload=True)
        self.assertEqual(value["usb_qualified_transfer_payload_bytes"], needed)


class PreflightTests(unittest.TestCase):
    def test_only_owner_readiness_may_hold_the_parent_before_the_run(self) -> None:
        owner_reasons = ("REMOTE_RESIDENT_OWNER_NOT_READY:HTP0", "RESIDENCY_TRANSITION_ABSENT", "ROUTE_NOT_QUALIFIED")
        self.assertEqual(remote_resident_route_blockers(owner_reasons), ())
        self.assertEqual(remote_resident_route_blockers(("RESIDENCY_TRANSITION_ABSENT",)),
                         ("RESIDENCY_TRANSITION_ABSENT",))
        self.assertEqual(remote_resident_route_blockers((*owner_reasons, "TRANSPORT_PROFILE_INCOMPLETE")),
                         ("TRANSPORT_PROFILE_INCOMPLETE",))
        self.assertEqual(remote_resident_route_blockers(("PLACEMENT_INFEASIBLE", "MEMORY_CAPACITY")),
                         ("MEMORY_CAPACITY", "PLACEMENT_INFEASIBLE"))

    def test_preflight_checks_the_route_and_its_transport(self) -> None:
        arm = pinned_arm(pin=False)
        at = arm.snapshot.captured_at_us
        candidates = arm.scheduler.generate_automated_candidates(
            arrival(arm, "preflight", at), arm.manifest.model_id, arm.snapshot, observed_at_us=at)
        row = SimpleNamespace(manifest=arm.manifest)
        checks = []
        _check_remote_resident_owner(checks, arm.catalog, row, candidates)
        self.assertEqual([(check.check_id.split(":")[0], check.status) for check in checks],
                         [("remote-resident-route", "PASS"), ("remote-resident-transport", "PASS")])
        without = SimpleNamespace(candidates=tuple(
            candidate for candidate in candidates.candidates
            if candidate.plan.execution_contract.remote_resident_ffn is None))
        small = replace(arm.catalog, placement_profile=replace(arm.catalog.placement_profile, links=tuple(
            replace(link, maximum_payload_bytes=61_440)
            if link.transport_generation.startswith("functionfs") else link
            for link in arm.catalog.placement_profile.links)))
        checks = []
        _check_remote_resident_owner(checks, small, row, without)
        self.assertEqual([check.status for check in checks], ["BLOCKED", "BLOCKED"])
        self.assertIn("none generated", checks[0].detail)
        self.assertIn(str(arm.manifest.embedding_length * 512 * 2), checks[1].detail)


class RunnerTests(unittest.TestCase):
    OWNER = RemoteResidentOwnerConfiguration("qwen3-32b-q4_0", 1)

    def test_arrivals_of_the_model_wait_for_ready_owners_bounded(self) -> None:
        pending = concurrent.futures.Future()
        gate = runner._RemoteResidentOwnerGate(self.OWNER, pending, time.monotonic_ns(), time.monotonic_ns())
        gate("overlay-0", "llama-3.2-1b-instruct-q4_0")
        self.assertIsNone(gate.ready_observed_at_us)
        with self.assertRaisesRegex(runner.UnifiedTraceError, "not READY within 1 s"):
            gate("q32-0", "qwen3-32b-q4_0")
        self.assertTrue(gate.timed_out)
        done = concurrent.futures.Future()
        done.set_result(object())
        gate = runner._RemoteResidentOwnerGate(self.OWNER, done, time.monotonic_ns(), time.monotonic_ns())
        gate("q32-0", "qwen3-32b-q4_0")
        gate("q32-1", "qwen3-32b-q4_0")
        self.assertEqual(gate.to_json()["first_request_id"], "q32-0")
        self.assertIsNotNone(gate.to_json()["ready_observed_at_us"])
        failed = concurrent.futures.Future()
        failed.set_exception(RuntimeError("owner preload failed"))
        gate = runner._RemoteResidentOwnerGate(self.OWNER, failed, time.monotonic_ns(), time.monotonic_ns())
        with self.assertRaisesRegex(RuntimeError, "owner preload failed"):
            gate("q32-0", "qwen3-32b-q4_0")

    def test_owner_preload_never_prepares_a_desktop_device(self) -> None:
        catalog = owner.owner_fixture().catalog

        def stage(*devices):
            return SimpleNamespace(helper_envelope=SimpleNamespace(preparation_transitions=(
                SimpleNamespace(prepares_device_ids=devices),)))

        runner._require_phone_only_preparation(stage("op15-phone"), catalog)
        cpu = next(key for key, row in catalog.placement_profile.devices.items() if row.kind == "cpu")
        with self.assertRaisesRegex(runner.UnifiedTraceError, "desktop transition"):
            runner._require_phone_only_preparation(stage("op15-phone", cpu), catalog)

    def ledger_inputs(self, executor_id="physical:hot:desktop:remote-owner"):
        group = RuntimeRemoteResidentFfn.from_json(route_fixture.declaration())
        manifest = route_fixture.MANIFEST
        configuration = {"executor_id": "physical:hot:desktop:remote-owner", "model_id": manifest.model_id,
                         "remote_resident_ffn": group.to_json()}
        demand = SimpleNamespace(kind="workspace", device_id="op15-phone", required_bytes=64 << 20)
        scheduler = SimpleNamespace(
            remote_resident_owner_configuration=lambda: configuration,
            runtime_ticket=lambda _request_id: SimpleNamespace(
                execution_plan=SimpleNamespace(memory_demands=(demand,))))
        proof = {"layer_mask": group.layer_mask, "omitted_bytes": group.omitted_bytes,
                 "schema": "scheduler-remote-resident-omission-v1", "unmapped_bytes": group.omitted_bytes,
                 "warmup": "validated"}
        launch = {"endpoint": "http://127.0.0.1:18080", "executor_id": "physical:hot:desktop:remote-owner",
                  "generation": 3, "host_available_bytes": 20 << 30, "observed_epoch_us": 5, "pid": 7,
                  "proof": proof, "vm_rss_bytes": 6 << 30}
        rig = SimpleNamespace(
            remote_resident_launches=(launch,),
            configuration=SimpleNamespace(host_memory_resource_id="host-ram", catalog=SimpleNamespace(
                composite_executor_by_id={"physical:hot:desktop:remote-owner": SimpleNamespace(
                    adapter_parameters={"phone_device_id": "op15-phone"})})))
        gate = runner._RemoteResidentOwnerGate(
            RemoteResidentOwnerConfiguration(manifest.model_id), concurrent.futures.Future(), 0, 0)
        results = [{"request_id": "r0", "model_id": manifest.model_id, "actual_executor_id": executor_id}]
        return scheduler, rig, {manifest.model_id: manifest}, results, gate, group

    def test_result_records_the_omission_proof_and_the_desktop_ram_saving(self) -> None:
        scheduler, rig, manifests, results, gate, group = self.ledger_inputs()
        value = runner._remote_resident_owner_result(scheduler, rig, manifests, results, gate)
        self.assertEqual(value["requests"], 1)
        ledger = value["ledger"]
        self.assertTrue(ledger["verified"])
        self.assertEqual(ledger["desktop_ram_saving_bytes"], group.omitted_bytes)
        account = ledger["launches"][0]["accounting"]
        self.assertEqual((account["phone_workspace_bytes"], account["proof"]["layer_mask"]),
                         (64 << 20, group.layer_mask))

    def test_result_refuses_a_pinned_request_on_another_route(self) -> None:
        scheduler, rig, manifests, results, gate, _group = self.ledger_inputs("physical:hot:desktop")
        with self.assertRaisesRegex(runner.UnifiedTraceError, "executed another route"):
            runner._remote_resident_owner_result(scheduler, rig, manifests, results, gate)


class RigLaunchRecordTests(unittest.TestCase):
    def test_reduced_launch_records_its_own_proof_and_host_memory(self) -> None:
        proof = SimpleNamespace(to_json=lambda: {"layer_mask": 3})
        rig = SimpleNamespace(epoch_ns=time.monotonic_ns())
        server = SimpleNamespace(pid=os.getpid(), remote_resident_proof=lambda: proof)
        record = RigTransitionMixin._remote_resident_launch_record(rig, server, "remote", "http://h:1", 2)
        self.assertEqual((record["executor_id"], record["generation"], record["proof"]), ("remote", 2, {"layer_mask": 3}))
        self.assertGreater(record["vm_rss_bytes"], 0)
        self.assertGreater(record["host_available_bytes"], 0)
        plain = SimpleNamespace(pid=os.getpid(), remote_resident_proof=lambda: None)
        self.assertIsNone(RigTransitionMixin._remote_resident_launch_record(rig, plain, "full", "http://h:1", 2))
        self.assertIsNone(RigTransitionMixin._remote_resident_launch_record(
            rig, SimpleNamespace(pid=1), "phone", "http://h:1", 2))


class PolicyGuardTests(unittest.TestCase):
    def test_joint_planner_prefill_yield_never_targets_a_remote_resident_joiner(self) -> None:
        controller = SimpleNamespace(_adaptive_decode_config=SimpleNamespace(server_policy_coherence=True))
        ticket = SimpleNamespace(dispatch_state="ACQUIRED", execution_plan=SimpleNamespace(
            desktop_placement_sha256="sha256:" + "c" * 64, helper_envelope=None,
            execution_contract=SimpleNamespace(execution_mode="desktop", remote_resident_ffn=object())))
        self.assertEqual(
            AdaptiveDecodeControlMixin.register_joint_join_prefill_yield(controller, ticket, at_us=1, expires_at_us=2),
            {"outcome": "JOINER_REMOTE_RESIDENT"})

    def test_no_adaptive_policy_covers_a_remote_resident_group(self) -> None:
        remote = SimpleNamespace(plan=SimpleNamespace(execution_contract=SimpleNamespace(
            remote_resident_ffn=object())))
        plain = SimpleNamespace(plan=SimpleNamespace(execution_contract=SimpleNamespace(remote_resident_ffn=None)))
        for candidate, parent in ((remote, plain), (plain, remote)):
            with self.subTest(candidate=candidate), self.assertRaisesRegex(AdaptiveDecodeError, "remote-resident"):
                _policy(candidate, parent, route_fixture.MANIFEST, 8)


if __name__ == "__main__":
    unittest.main()
