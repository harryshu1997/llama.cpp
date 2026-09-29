"""Elastic phones, slice 2: device quarantine, readmission and the co-helper membership lifecycle.

Recorded tests, no hardware: the coherent server of test_per_device_policies (primary "op15",
co-helper "pixel"), the synthetic two-phone catalog of two_phone_harness, fake adb workers and a
fake scheduler behind the rig's membership probe."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from research_dev.scheduler import (
    DEVICE_MEMBERSHIP_EVENT_KINDS,
    UnifiedScheduleError,
    UnifiedScheduler,
    elastic_phones_configuration,
)
from research_dev.scheduler._internal.adaptive_decode_contracts import AdaptiveDecodeError
from research_dev.scheduler._internal.adaptive_decode_ops.coherence import (
    DEVICE_QUARANTINED,
    DEVICE_SET_FAILED,
    server_verdict,
)
from research_dev.scheduler._internal.adaptive_decode_planning import adaptive_decode_policies
from research_dev.scheduler._internal.adaptive_decode_state import _AdaptiveServerPolicy
from research_dev.scheduler._internal.runtime_controller import RuntimeController
from research_dev.scheduler.adapters import heterogeneous_rig as rig_module
from research_dev.scheduler.adapters.co_helper_lifecycle import CoHelperLifecycle, IdleCoHelperStopPolicy
from research_dev.scheduler.adapters.contracts import PhysicalAdapterError
from research_dev.scheduler.adapters.energy import PhoneActivityIntervalTracker
from research_dev.scheduler.adapters.heterogeneous_rig import HeterogeneousPhysicalRig
from research_dev.scheduler.adapters.phone_tcp_session import (
    CONNECTED_MARKER,
    READY_MARKER,
    AdbTcpPhoneWorkerSession,
    AdbTcpWorkerConfiguration,
)
from research_dev.scheduler.campaigns.burstgpt import arguments, preflight
from research_dev.scheduler.campaigns.burstgpt.tools import inject_helper_loss as tool
from research_dev.scheduler.configuration.campaign import CampaignManifest
from research_dev.scheduler.configuration.common import SchedulerConfigurationError

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
import two_phone_harness as h  # noqa: E402
from test_per_device_policies import (  # noqa: E402
    C, CO_HELPER, HOST, P, PC, PRIMARY, _DeviceSetCase,
)

ELASTIC = elastic_phones_configuration({
    "drop_recovery": True, "join": True, "join_probe_interval_s": 1,
    "readmission_cooldown_s": 0, "max_readmissions_per_device": 2,
})
IDENTITY = "sha256:" + "9" * 64


# ---- configuration -------------------------------------------------------------------------------

class ElasticConfigurationTests(unittest.TestCase):
    def test_normalized_form_spells_out_every_field(self):
        self.assertIsNone(elastic_phones_configuration(None))
        self.assertEqual(dict(elastic_phones_configuration({"join": True})), {
            "drop_recovery": False, "join": True, "join_probe_interval_s": 10,
            "max_readmissions_per_device": 3, "readmission_cooldown_s": 60})
        for value in ({}, {"drop_recovery": False, "join": False}, {"join": 1}, {"join": True, "extra": 1},
                      {"join": True, "join_probe_interval_s": 0}, {"join": True, "readmission_cooldown_s": -1},
                      {"join": True, "max_readmissions_per_device": True}, [], "join"):
            with self.subTest(value=value), self.assertRaises(SchedulerConfigurationError):
                elastic_phones_configuration(value)

    def _manifest(self, **extra):
        with tempfile.TemporaryDirectory() as directory:
            return CampaignManifest.from_json({
                "campaign_id": "elastic", "energy_attribution_kind": "diagnostic",
                "evidence_manifest_path": "evidence.json", "maximum_latency_ppm": 1_250_000,
                "models_manifest_path": "models.json", "rig_manifest_path": "rig.json",
                "schema": "research-scheduler-campaign-v1", "selection_mode": "adaptive-decode",
                "trace": {"large_requests_path": "l.jsonl", "overlay_requests_path": "o.jsonl",
                          "trace_manifest_path": "t.json"},
                **extra,
            }, Path(directory))

    def test_campaign_field_is_absent_unless_declared(self):
        plain = self._manifest()
        self.assertIsNone(plain.elastic_phones)
        self.assertNotIn("elastic_phones", plain.to_json())
        elastic = self._manifest(elastic_phones={"drop_recovery": True, "join": True})
        self.assertTrue(elastic.elastic_phones["join"])
        self.assertEqual(elastic.to_json()["elastic_phones"]["readmission_cooldown_s"], 60)
        with self.assertRaises(SchedulerConfigurationError):
            self._manifest(elastic_phones={"join": "yes"})

    def test_runner_and_preflight_arguments_parse_the_same_object(self):
        parsed = arguments.elastic_phones_json('{"join": true, "readmission_cooldown_s": 5}')
        self.assertEqual(parsed["readmission_cooldown_s"], 5)
        self.assertIs(type(parsed), dict)
        for text in ("null", "[]", "{", '{"join": false}'):
            with self.subTest(text=text), self.assertRaises((argparse.ArgumentTypeError, ValueError)):
                arguments.elastic_phones_json(text)
        parser = arguments._build_parser()
        action = next(row for row in parser._actions if "--elastic-phones-json" in row.option_strings)
        self.assertEqual(action.dest, "elastic_phones")
        self.assertIsNone(action.default)


class ElasticLaunchCommandTests(unittest.TestCase):
    """launch.py passes the campaign field to the runner and the preflight, only when declared."""

    def test_commands_carry_the_flag_only_when_declared(self):
        from test_two_phone_helpers import CampaignConfigurationTests, two_phone_rig_json
        from research_dev.scheduler.campaigns.burstgpt import launch
        from research_dev.scheduler.config import load_scheduler_configuration

        case = CampaignConfigurationTests()
        case.setUp()
        self.addCleanup(case.tearDown)
        path = case._write(two_phone_rig_json(), None)
        plain = load_scheduler_configuration(path, environ={})
        arguments_of = lambda configuration: launch.preflight_command(  # noqa: E731
            configuration, catalog_path=case.root / "c.json", normal_usb_receipt_path=case.root / "u.json",
            output_path=case.root / "out.json")
        self.assertNotIn("--elastic-phones-json", arguments_of(plain))
        campaign = json.loads(path.read_text())
        campaign["elastic_phones"] = {"join": True, "drop_recovery": True}
        path.write_text(json.dumps(campaign))
        elastic = load_scheduler_configuration(path, environ={})
        command = arguments_of(elastic)
        value = json.loads(command[command.index("--elastic-phones-json") + 1])
        self.assertEqual(value, dict(elastic.campaign.elastic_phones))


class ElasticPreflightTests(unittest.TestCase):
    """An absent co-helper is recorded, not fatal, only when it may join later."""

    def _rows(self, helper_passed):
        row = lambda name, passed: type("Row", (), {"name": name, "passed": passed, "detail": "d"})()  # noqa: E731
        return (row("phone-usb-port:primary-phone", True), row("phone-usb-port:pixel10pro-phone", helper_passed),
                row("phone-usb-topology", helper_passed))

    def _args(self, elastic):
        from test_two_phone_helpers import OP15_SERIAL, pixel_helper_json
        return argparse.Namespace(helper_phone=[json.dumps(pixel_helper_json())], phone_usb_serial=OP15_SERIAL,
                                  minimum_usb_speed_mbps=5000, elastic_phones=elastic)

    def test_absent_helper_is_recorded_under_join_only(self):
        observed = {"primary-phone": SimpleNamespace(root_port="2-1")}
        for elastic, status in ((None, "BLOCKED"), (elastic_phones_configuration({"drop_recovery": True}), "BLOCKED"),
                                (ELASTIC, "PASS")):
            with self.subTest(elastic=None if elastic is None else dict(elastic)), mock.patch.object(
                    preflight, "check_usb_topology", return_value=(self._rows(False), observed)):
                checks = preflight._helper_phone_checks(self._args(elastic))
            by_id = {row.to_json()["check_id"]: row.to_json() for row in checks}
            self.assertEqual(by_id["phone-usb-port:pixel10pro-phone"]["status"], status)
            self.assertEqual(by_id["phone-usb-topology"]["status"], status)
            # without evidence the two-phone dispatch stays blocked whatever the membership
            self.assertEqual(by_id["two-phone-dispatch"]["status"], "BLOCKED")

    def test_present_helper_with_a_wrong_port_still_fails(self):
        observed = {"primary-phone": SimpleNamespace(root_port="2-1"),
                    "pixel10pro-phone": SimpleNamespace(root_port="2-9")}
        with mock.patch.object(preflight, "check_usb_topology", return_value=(self._rows(False), observed)):
            checks = preflight._helper_phone_checks(self._args(ELASTIC))
        self.assertEqual({row.to_json()["check_id"]: row.to_json()["status"] for row in checks}
                         ["phone-usb-port:pixel10pro-phone"], "BLOCKED")


# ---- adaptive coherence --------------------------------------------------------------------------

class CoherenceQuarantineTests(_DeviceSetCase):
    def _second_group(self, server):
        group = _AdaptiveServerPolicy("other-owner", self.baseline, 0, device_sets=server.group().device_sets,
                                      verdicts=((1, self.both), (2, self.primary)), proposal=self.helper)
        group.policy = self.both
        server.controller._server_policies[("other", "group", None, "sha256:" + "6" * 64)] = group
        return group

    def test_quarantine_drops_the_device_in_every_group_and_falls_back_to_the_primary(self):
        server = self.server()
        group = self.qualify_both_at_batch_one(server)
        other = self._second_group(server)
        self.assertEqual(server.running(), PC)
        self.assertTrue(server.controller.quarantine_device(CO_HELPER, reason="HELPER_LOST", at_us=server.now))
        self.assertFalse(server.controller.quarantine_device(CO_HELPER, reason="HELPER_LOST", at_us=server.now))
        dropped = ((0, PC, DEVICE_QUARANTINED), (0, C, DEVICE_QUARANTINED))
        for row in (group, other):
            self.assertEqual(row.device_drops, dropped)
            self.assertTrue(all(CO_HELPER not in policy_devices for policy_devices in (
                tuple(device for device, _mask in verdict.device_layer_masks) for _batch, verdict in row.verdicts)))
        self.assertEqual(other.verdicts, ((2, self.primary),))
        self.assertIsNone(other.proposal)
        self.assertEqual(group.policy, self.baseline)
        # the live session eliminates the device's policies at once
        reasons = server.controller._sessions["request-a"].eliminated_policy_reasons
        self.assertEqual({reasons[self.both.policy_hash], reasons[self.helper.policy_hash]}, {DEVICE_QUARANTINED})
        del server.controls[:]
        server.tick()
        self.assertEqual(server.controls, [("request-a", P)])
        self.assertEqual(server_verdict(group, 1), self.primary)
        for _ in range(3):
            server.tick()
        self.assertEqual(server.running(), P)
        self.assertEqual(server.controller.quarantined_devices, {CO_HELPER: "HELPER_LOST"})

    def test_readmission_restores_the_sets_and_the_probe_decides_again(self):
        server = self.server()
        group = self.qualify_both_at_batch_one(server)
        server.controller.quarantine_device(CO_HELPER, reason="HELPER_LOST", at_us=server.now)
        server.tick()
        self.assertEqual(server_verdict(group, 1), self.primary)
        self.assertTrue(server.controller.readmit_device(CO_HELPER))
        self.assertFalse(server.controller.readmit_device(CO_HELPER))
        self.assertEqual(group.device_drops, ())
        self.assertEqual(server.controller._sessions["request-a"].eliminated_policy_reasons, {})
        # the primary verdict stays; the set that adds the co-helper challenges it on evidence
        del server.controls[:]
        server.tick()
        self.assertEqual(server.controls, [("request-a", PC)])
        for _ in range(3):
            server.tick()
        self.assertEqual(server_verdict(group, 1), self.both)
        self.assertEqual(server.running(), PC)

    def test_readmission_clears_failures_and_the_host_verdict_they_decided(self):
        server = self.server()
        group = self.qualify_both_at_batch_one(server)
        # the co-helper fails (the failed-window path) and is then quarantined by the rig
        server.tick(failure="helper pixel: timeout")
        server.controller.quarantine_device(CO_HELPER, reason="HELPER_LOST", at_us=server.now)
        server.tick(failure="helper op15: timeout")
        for _ in range(2):
            server.tick()
        self.assertEqual(server_verdict(group, 1), self.baseline)
        self.assertIn((0, P, DEVICE_SET_FAILED), group.device_drops)
        server.controller.readmit_device(CO_HELPER)
        # the co-helper's own rows are gone; the primary's failure still drops its superset and the
        # host verdict the failures decided is withdrawn, so the co-helper alone is probed again
        self.assertEqual(set(group.device_drops), {(0, P, DEVICE_SET_FAILED), (0, PC, DEVICE_SET_FAILED)})
        self.assertIsNone(server_verdict(group, 1))
        for _ in range(4):
            server.tick()
        self.assertEqual(server.running(), C)

    def test_groups_created_after_the_quarantine_start_without_the_device(self):
        server = self.server()
        server.controller.quarantine_device(CO_HELPER, reason="DEVICE_ABSENT_AT_START", at_us=0)
        server.start("request-a", 1)
        group = server.group()
        self.assertEqual(group.device_drops, ((0, PC, DEVICE_QUARANTINED), (0, C, DEVICE_QUARANTINED)))
        for _ in range(6):
            server.tick()
        self.assertEqual(server_verdict(group, 1), self.primary)
        self.assertEqual({devices for _request, devices in server.controls} - {HOST}, {P})
        # the primary fails too: with the co-helper away no phone is left
        server.tick(failure="helper op15: timeout")
        for _ in range(2):
            server.tick()
        self.assertEqual(server_verdict(group, 1), self.baseline)
        self.assertEqual(server.running(), HOST)

    def test_membership_survives_a_checkpoint_restore(self):
        server = self.server()
        group = self.qualify_both_at_batch_one(server)
        checkpoint = server.controller.checkpoint()
        server.controller.quarantine_device(CO_HELPER, reason="HELPER_LOST", at_us=server.now)
        server.controller.restore(checkpoint)
        restored = server.group()
        self.assertEqual(restored.device_drops, ())
        server.tick()
        self.assertEqual(restored.device_drops, ((0, PC, DEVICE_QUARANTINED), (0, C, DEVICE_QUARANTINED)))
        self.assertEqual(server.running(), P)
        self.assertIsNot(group, restored)

    def test_request_local_sessions_never_probe_the_quarantined_device(self):
        self.config = replace(self.config, server_policy_coherence=False, maximum_probe_candidates=4)
        server = self.server()
        server.controller.quarantine_device(CO_HELPER, reason="DEVICE_ABSENT_AT_START", at_us=0)
        server.start("request-a", 1)
        session = server.controller._sessions["request-a"]
        self.assertEqual({session.eliminated_policy_reasons.get(row.policy_hash) for row in (self.both, self.helper)},
                         {DEVICE_QUARANTINED})
        self.assertTrue(session.probe_candidates)
        self.assertTrue(all(CO_HELPER not in tuple(device for device, _mask in row.device_layer_masks)
                            for row in session.probe_candidates))
        for _ in range(8):
            server.tick()
        self.assertNotIn(PC, {devices for _request, devices in server.controls})
        self.assertNotIn(C, {devices for _request, devices in server.controls})

    def test_a_quarantined_primary_opens_the_co_helper_alone(self):
        server = self.server()
        group = self.qualify_both_at_batch_one(server)
        # a batch-specific drop of the primary alone must not hide its quarantine
        group.device_drops += ((1, P, "SERVER_DEVICE_SET_NOT_IMPROVED"),)
        server.controller.quarantine_device(PRIMARY, reason="HELPER_LOST", at_us=server.now)
        for _ in range(4):
            server.tick()
        self.assertEqual(server.running(), C)
        self.assertEqual(server_verdict(group, 1), self.helper)

    def test_late_helper_candidates_are_eliminated_on_attachment(self):
        from test_adaptive_decode import ARTIFACT, PLAN
        controller = self.server().controller
        controller.start(
            request_id="late", ticket_id="late:attempt:0", model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN, baseline=self.baseline, candidates=(), output_tokens=400,
            context_length=64, active_batch=1, deadline_us=10**12, slot_id=1, first_token_index=1,
            first_token_at_us=1_000, config=self.config, helper_evidence_state="LEARNING", helper_available=False)
        controller.quarantine_device(CO_HELPER, reason="HELPER_LOST", at_us=5)
        # attachment replaces the candidates and clears eliminations; the quarantine is re-applied
        controller.helper_ready("late", phone_layout_generation=1, phone_layout_geometry_sha256="sha256:" + "7" * 64,
                                candidates=self.candidates, component_capability_sha256=PLAN,
                                helper_evidence_state="LEARNING")
        self.assertEqual(controller._sessions["late"].eliminated_policy_reasons, {
            self.both.policy_hash: DEVICE_QUARANTINED, self.helper.policy_hash: DEVICE_QUARANTINED})

    def test_invalid_membership_arguments_fail_closed(self):
        controller = self.server().controller
        for arguments_ in (("", "HELPER_LOST", 0), (CO_HELPER, "", 0), (CO_HELPER, "HELPER_LOST", -1),
                           (CO_HELPER, "HELPER_LOST", 1.0)):
            with self.subTest(arguments=arguments_), self.assertRaises(AdaptiveDecodeError):
                controller.quarantine_device(arguments_[0], reason=arguments_[1], at_us=arguments_[2])
        self.assertEqual(controller.quarantined_devices, {})


class SinglePhoneQuarantineTests(_DeviceSetCase):
    def test_quarantine_leaves_single_phone_groups_unchanged(self):
        from test_adaptive_decode import policy
        self.candidates = (policy("phone-full", 1000, (2, 3)),)
        server = _DeviceSetCase.server(self)
        server.energies, server.latencies = {((), 1): 100}, {((), 1): 1_000}
        server.start("request-a", 1)
        server.controller.quarantine_device("op15", reason="HELPER_LOST", at_us=0)
        self.assertEqual(server.group().device_drops, ())
        self.assertEqual(server.controller._sessions["request-a"].eliminated_policy_reasons, {})


# ---- runtime controller and the unified scheduler -------------------------------------------------

class RuntimeControllerQuarantineTests(unittest.TestCase):
    def test_device_resources_are_withheld_until_readmission(self):
        controller = RuntimeController()
        binding = SimpleNamespace(route_id="phone-route", resource_ids=("op15-htp", "cpu"), ready=True,
                                  queueable=True, eligibility_reasons=())
        with mock.patch("research_dev.scheduler._internal.runtime_controller_ops.admission.replace",
                        side_effect=lambda row, **changes: SimpleNamespace(**{**vars(row), **changes})):
            self.assertTrue(controller.effective_bindings((binding,))[0].ready)
            checkpoint = controller.checkpoint()
            self.assertTrue(controller.quarantine_device(
                "op15-phone", resource_ids=("op15-htp",), reason="HELPER_LOST", at_us=5))
            self.assertFalse(controller.quarantine_device(
                "op15-phone", resource_ids=("op15-htp",), reason="HELPER_LOST", at_us=6))
            controller.restore(checkpoint)
            effective = controller.effective_bindings((binding,))[0]
            self.assertEqual((effective.ready, effective.eligibility_reasons), (False, ("RESOURCE_QUARANTINED",)))
        self.assertEqual(controller.quarantined_devices, frozenset({"op15-phone"}))
        self.assertEqual(controller.snapshot()["quarantined_devices"]["op15-phone"]["action"], "device_lost")
        self.assertIn("op15-htp", controller.snapshot()["quarantined_resources"])
        self.assertTrue(controller.readmit_device("op15-phone"))
        self.assertFalse(controller.readmit_device("op15-phone"))
        self.assertFalse(controller.resources_are_quarantined(("op15-htp",)))
        self.assertNotIn("quarantined_devices", controller.snapshot())
        self.assertEqual(controller.checkpoint(), checkpoint)


class SchedulerMembershipTests(unittest.TestCase):
    def setUp(self):
        scope = h.TemporaryModel()
        self.model = scope.__enter__()
        self.addCleanup(scope.__exit__)
        self.catalog = h.runtime_catalog(self.model, declaration=h.co_helpers())
        self.scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        self.scheduler.register_runtime_capabilities(self.catalog)
        self.scheduler.register_model_manifest(self.model)

    def test_quarantine_and_readmission_are_idempotent_events(self):
        scheduler = self.scheduler
        scheduler.quarantine_device(h.PIXEL, reason="DEVICE_ABSENT_AT_START", at_us=10)
        scheduler.quarantine_device(h.PIXEL, reason="HELPER_LOST", at_us=20)
        scheduler.quarantine_device(h.OP15, reason="HELPER_LOST", at_us=30)
        self.assertEqual(scheduler.quarantined_devices(), {h.PIXEL: "DEVICE_ABSENT_AT_START", h.OP15: "HELPER_LOST"})
        devices = scheduler.runtime_controller_snapshot()["quarantined_devices"]
        # a static co-helper withholds no resource (they also bind the primary-only sets)
        self.assertEqual(devices[h.PIXEL]["resource_ids"], [])
        self.assertEqual(devices[h.OP15]["resource_ids"], ["op15-functionfs", "op15-htp"])
        scheduler.readmit_device(h.PIXEL, at_us=40, identity_sha256=IDENTITY)
        scheduler.readmit_device(h.PIXEL, at_us=50, identity_sha256=IDENTITY)
        events = scheduler.device_membership_events()
        self.assertEqual([(row["kind"], row["device_id"], row["at_us"]) for row in events], [
            ("DEVICE_ABSENT_AT_START", h.PIXEL, 10), ("DEVICE_QUARANTINED", h.OP15, 30),
            ("DEVICE_READMITTED", h.PIXEL, 40)])
        self.assertEqual(events[2]["identity_sha256"], IDENTITY)
        self.assertTrue({row["kind"] for row in events} <= set(DEVICE_MEMBERSHIP_EVENT_KINDS))
        self.assertEqual(scheduler.quarantined_devices(), {h.OP15: "HELPER_LOST"})

    def test_invalid_membership_calls_are_refused(self):
        scheduler = self.scheduler
        for device in ("desk-cpu", "unknown-phone", ""):
            with self.subTest(device=device), self.assertRaises(UnifiedScheduleError):
                scheduler.quarantine_device(device, reason="HELPER_LOST", at_us=0)
        with self.assertRaises(UnifiedScheduleError):
            scheduler.readmit_device(h.PIXEL, at_us=0, identity_sha256="sha256:bad")
        with self.assertRaises(UnifiedScheduleError):
            scheduler.quarantine_device(h.PIXEL, reason="HELPER_LOST", at_us=-1)
        self.assertEqual(scheduler.device_membership_events(), ())
        with self.assertRaises(UnifiedScheduleError):
            UnifiedScheduler.for_runtime_discovery("enforce").quarantine_device(h.PIXEL, reason="X", at_us=0)


# ---- route generation: an absent co-helper keeps the primary-only sets -----------------------------

def membership_snapshot(model, catalog, state="ABSENT"):
    value = h.snapshot(model, catalog, pixel_hot=False)
    executors = dict(value.executors)
    executors["physical:" + h.PIXEL] = replace(executors["physical:" + h.PIXEL], ready=False, free_slots=0)
    return replace(value, executors=executors, telemetry_observations={h.PIXEL: {
        "failure_reason": "co-helper pixel-phone is " + state, "membership": state, "source": "helper-membership",
        "valid": False, "validity": "UNAVAILABLE"}})


class AbsentCoHelperRouteTests(unittest.TestCase):
    def setUp(self):
        scope = h.TemporaryModel()
        self.model = scope.__enter__()
        self.addCleanup(scope.__exit__)
        self.catalog = h.runtime_catalog(self.model, declaration=h.co_helpers())

    def _scheduler(self):
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(self.catalog)
        scheduler.register_model_manifest(self.model)
        return scheduler

    def test_absent_co_helper_keeps_the_envelope_and_its_primary_only_sets(self):
        for state in ("ABSENT", "QUARANTINED"):
            with self.subTest(state=state):
                values = self._scheduler().generate_automated_candidates(
                    h.request("absent"), self.model.model_id, membership_snapshot(self.model, self.catalog, state))
                split = next(row for row in values.candidates if row.binding.executor_id.endswith(":operator_split"))
                self.assertIn("CO_HELPER_UNAVAILABLE", split.rejection_reasons)
                self.assertNotIn("PHONE_TELEMETRY_UNAVAILABLE", split.rejection_reasons)
                self.assertNotIn("RESIDENCY_TRANSITION_ABSENT", split.rejection_reasons)
                self.assertFalse(split.admitted)
                _baseline, policies, envelope = adaptive_decode_policies(values, self.model, self.catalog, 30)
                self.assertIsNotNone(envelope)
                self.assertIn((h.OP15,), {tuple(device for device, _mask in row.device_layer_masks)
                                          for row in policies})

    def test_an_unmarked_unavailable_co_helper_still_blocks_the_route(self):
        value = membership_snapshot(self.model, self.catalog)
        row = dict(value.telemetry_observations[h.PIXEL])
        del row["membership"]
        values = self._scheduler().generate_automated_candidates(
            h.request("unmarked"), self.model.model_id, replace(value, telemetry_observations={h.PIXEL: row}))
        split = next(row for row in values.candidates if row.binding.executor_id.endswith(":operator_split"))
        self.assertIn("PHONE_TELEMETRY_UNAVAILABLE", split.rejection_reasons)
        self.assertEqual(adaptive_decode_policies(values, self.model, self.catalog, 30)[1], ())

    def test_adaptive_ticket_is_admitted_while_the_co_helper_is_quarantined(self):
        scheduler = self._scheduler()
        scheduler.quarantine_device(h.PIXEL, reason="DEVICE_ABSENT_AT_START", at_us=0)
        ticket = scheduler.submit_automated_request(
            h.request("absent-adaptive"), self.model.model_id, membership_snapshot(self.model, self.catalog),
            selection_mode="adaptive-decode")
        ticket = scheduler.wait_runtime_request("absent-adaptive", time.monotonic_ns() - ticket.decision.start_us * 1000)
        self.assertEqual(ticket.execution_plan.execution_contract.execution_mode, "adaptive-split")


# ---- adb-tcp worker session with fake adb -----------------------------------------------------------

WORKER = "/data/local/tmp/w/llama-ffn-split-worker"
SHARD = "/data/local/tmp/w/shard.gguf"


class _Process:
    def __init__(self, code=None):
        self.code, self.terminated = code, False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated, self.code = True, -15

    def wait(self, timeout=None):
        return self.code

    @property
    def returncode(self):
        return self.code


class _FakeAdb:
    def __init__(self):
        self.commands, self.pids, self.forwards, self.failing = [], [], set(), set()
        self.boot, self.on_kill = "boot-1", None

    def run(self, argv, **_options):
        words = argv[5:]
        self.commands.append(" ".join(words))
        key = words[0] if words[0] != "shell" else words[1].split()[0]
        if key in self.failing:
            raise subprocess.CalledProcessError(1, argv, "", "error: device not found")
        if words[:2] == ["shell", "ps -A -o PID,ARGS"]:
            out = "PID ARGS\n" + "".join(f"{pid} {WORKER} -m {SHARD}\n" for pid in self.pids)
        elif words[:1] == ["shell"] and words[1].startswith("sha256sum"):
            out = "".join(f"{'a' * 64}  {path}\n" for path in (WORKER, SHARD))
        elif words[:1] == ["shell"] and "boot_id" in words[1]:
            out = self.boot + "\n"
        elif words[:1] == ["shell"] and words[1].startswith("kill -TERM"):
            self.pids.remove(int(words[1].split()[-1]))
            if self.on_kill is not None:
                self.on_kill(int(words[1].split()[-1]))
            out = ""
        elif words[:2] == ["forward", "--no-rebind"]:
            self.forwards.add("SERIAL1 tcp:40317 tcp:26990")
            out = "40317\n"
        elif words[:2] == ["forward", "--list"]:
            out = "".join(line + "\n" for line in sorted(self.forwards))
        elif words[:2] == ["forward", "--remove"]:
            self.forwards.clear()
            out = ""
        else:
            out = ""
        return subprocess.CompletedProcess(argv, 0, out, "")


class AdbTcpSessionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.adb = _FakeAdb()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.clock = [0.0]
        self.process = _Process()
        self.on_popen = None

    def _session(self, *, ready=True, helper=False):
        """``helper``: the worker of the two_phone_harness co-helper declaration (h.co_helpers())."""
        configuration = AdbTcpWorkerConfiguration(
            device_id=h.PIXEL if helper else "pixel10pro-phone", serial=h.PIXEL_SERIAL if helper else "SERIAL1",
            adb_port=5037, adb_path=Path("/usr/bin/adb"),
            worker_path=WORKER, library_directories=("/data/local/tmp/w",), shard_path=SHARD,
            artifact_sha256="sha256:" + "a" * 64, layer_mask=h.PIXEL_MASK if helper else 0b1100, n_embd=64,
            columns=128, column_quantum=32 if helper else 64, max_tokens=4, swiglu=True, backend="CPU",
            phone_port=26990, forward_port=26991 if helper else 0, launch_timeout_s=5.0,
            expected_sha256_by_path={WORKER: "sha256:" + "a" * 64, SHARD: "sha256:" + "a" * 64})

        def popen(command, *, stdout, **_options):
            self.adb.pids[:] = [4242]
            if self.on_popen is not None:
                self.on_popen()
            if ready:
                stdout.write(READY_MARKER + "CPU layers=2\n")
                stdout.flush()
            return self.process

        def sleep(seconds):
            self.clock[0] += seconds

        return AdbTcpPhoneWorkerSession(configuration, run=self.adb.run, popen=popen, sleep=sleep,
                                        clock=lambda: self.clock[0])

    def test_failed_start_leaves_the_session_inactive(self):
        """Elastic phones (``release_failed_start``): the worker a failed start launched is SIGTERMed on
        the preflighted boot before the session goes inactive, whether it never became ready or it
        reached READY and the forward failed."""
        session = self._session(ready=False)
        self.process.code = 1
        with self.assertRaisesRegex(PhysicalAdapterError, "did not become ready"):
            session.start(self.root / "worker.log", release_failed_start=True)
        self.assertFalse(session.active)
        self.assertIsNone(session._boot_id)
        self.assertIn("shell kill -TERM 4242", self.adb.commands)
        self.assertEqual(self.adb.pids, [])
        with self.assertRaisesRegex(PhysicalAdapterError, "not active"):
            session.transport_parameters()
        # READY reached, then the forward failed: the worker is released, the local client ended
        self.adb.commands.clear()
        self.process = _Process()
        session = self._session()
        self.adb.failing.add("forward")
        with self.assertRaises(subprocess.CalledProcessError):
            session.start(self.root / "worker-2.log", release_failed_start=True)
        self.assertFalse(session.active)
        self.assertIn("shell kill -TERM 4242", self.adb.commands)
        self.assertEqual(self.adb.pids, [])
        self.assertTrue(self.process.terminated)
        self.assertFalse(any("-9" in row or "KILL" in row for row in self.adb.commands))

    def test_failed_start_signals_nothing_after_a_reboot(self):
        session = self._session()
        self.adb.failing.add("forward")

        def reboot():
            self.adb.boot = "boot-2"

        self.on_popen = reboot
        with self.assertRaises(subprocess.CalledProcessError):
            session.start(self.root / "worker.log", release_failed_start=True)
        self.assertFalse(session.active)
        self.assertFalse(any("kill" in row for row in self.adb.commands))

    def test_failed_start_keeps_todays_session_without_elastic_phones(self):
        """Flag absent: a failed start leaves the session active, the co-helper counts as started and the
        trace's stop policy SIGTERMs the worker at end_trace (the behaviour before elastic phones)."""
        session = self._session(helper=True)
        self.adb.failing.add("forward")
        with self.assertRaises(subprocess.CalledProcessError):
            session.start(self.root / "worker.log")
        self.assertTrue(session.active)
        self.assertEqual(session._boot_id, "boot-1")
        self.assertFalse(any("kill" in row for row in self.adb.commands))
        self.assertEqual(self.adb.pids, [4242])
        # the same failure through the static lifecycle: started, then stopped by the stop policy
        self.adb.commands.clear()
        self.adb.pids.clear()
        self.process = _Process()
        session = self._session(helper=True)
        lifecycle = CoHelperLifecycle(h.co_helpers(), {h.PIXEL: session}, IdleCoHelperStopPolicy())
        with self.assertRaises(subprocess.CalledProcessError):
            lifecycle.start_trace(self.root)
        self.assertEqual(lifecycle.started, [h.PIXEL])
        self.adb.failing.clear()
        self.adb.on_kill = lambda _pid: setattr(self.process, "code", 0)
        (receipt,) = lifecycle.end_trace({})
        self.assertEqual((receipt["phase"], receipt["signalled"]), ("stop", True))
        self.assertIn("shell kill -TERM 4242", self.adb.commands)
        self.assertEqual((lifecycle.started, self.adb.pids), ([], []))

    def test_join_releases_the_worker_its_failed_start_orphaned(self):
        """Elastic join: a worker that outlived a failed start (its SIGTERM could not reach the phone)
        makes every later preflight refuse "already runs"; the join releases it (same boot, recorded pid)
        and preflights once more. A foreign worker is never signalled."""
        session = self._session(helper=True)
        lifecycle = CoHelperLifecycle(h.co_helpers(), {h.PIXEL: session}, IdleCoHelperStopPolicy(),
                                      identity_checks={h.PIXEL: lambda: IDENTITY})
        self.adb.failing.update({"forward", "kill"})
        receipts = lifecycle.start_trace(self.root, tolerate_absent=True)
        self.assertEqual(receipts[-1]["kind"], "DEVICE_ABSENT_AT_START")
        self.assertEqual((lifecycle.absent, session.active, self.adb.pids), ({h.PIXEL}, False, [4242]))
        self.adb.failing.clear()
        # a foreign worker: refused, nothing signalled
        self.adb.pids[:] = [5555]
        with self.assertRaisesRegex(PhysicalAdapterError, "already runs"):
            lifecycle.join(h.PIXEL, self.root / "join1.log")
        self.assertEqual(self.adb.pids, [5555])
        self.assertEqual(lifecycle.absent, {h.PIXEL})
        # the orphan of our failed start: released, then the join proceeds
        self.adb.pids[:] = [4242]
        self.process = _Process()
        receipts = lifecycle.join(h.PIXEL, self.root / "join2.log")
        self.assertEqual([row.get("phase") or row.get("kind") for row in receipts],
                         ["release_orphan", "preflight", "launch", "JOINED"])
        self.assertEqual(receipts[0]["signalled_pids"], [4242])
        self.assertEqual((lifecycle.started, lifecycle.absent, session.active), ([h.PIXEL], set(), True))
        self.assertFalse(any("-9" in row or "KILL" in row for row in self.adb.commands))

    def test_client_exit_is_the_irreversible_loss_signal(self):
        session = self._session()
        self.assertFalse(session.client_exited())
        session.start(self.root / "worker.log")
        self.assertFalse(session.client_exited())
        self.adb.pids.clear()  # an adb view without the worker: not alive, but not proven gone
        self.assertEqual((session.alive(), session.client_exited()), (False, False))
        self.process.code = 0  # su and the local adb client exit with the worker
        self.assertEqual((session.alive(), session.client_exited()), (False, True))
        session.release_lost()
        self.assertFalse(session.client_exited())

    def test_alive_needs_process_pid_and_forward(self):
        session = self._session()
        self.assertFalse(session.alive())
        session.start(self.root / "worker.log")
        self.assertTrue(session.alive())
        self.adb.forwards.clear()
        self.assertFalse(session.alive())
        self.adb.forwards.add("SERIAL1 tcp:40317 tcp:26990")
        self.adb.failing.add("ps")
        self.assertFalse(session.alive())
        self.adb.failing.clear()
        self.adb.pids.clear()
        self.assertFalse(session.alive())

    def test_fault_injection_is_sigterm_only_and_needs_authorization(self):
        session = self._session()
        session.start(self.root / "worker.log")
        for authorized, elastic in ((False, ELASTIC), (True, None), ("yes", ELASTIC)):
            with self.subTest(authorized=authorized), self.assertRaisesRegex(PhysicalAdapterError, "authorization"):
                session.terminate_for_fault_injection(authorized=authorized, elastic_phones=elastic)
        receipt = session.terminate_for_fault_injection(authorized=True, elastic_phones=ELASTIC)
        self.assertEqual(receipt.details["signal"], "TERM")
        self.assertEqual(session.fault_injections[0]["worker_pids"], [4242])
        signals = [row for row in self.adb.commands if "kill" in row]
        self.assertEqual(signals, ["shell kill -TERM 4242"])
        self.assertFalse(any("-9" in row or "KILL" in row for row in self.adb.commands))

    def test_release_of_a_lost_worker_is_best_effort(self):
        session = self._session()
        session.start(self.root / "worker.log")
        self.adb.failing.update({"forward"})
        receipt = session.release_lost()
        self.assertFalse(session.active)
        self.assertEqual(receipt.details["signalled_pids"], [4242])
        self.assertFalse(receipt.details["forward_removed"])
        self.assertTrue(receipt.details["errors"])
        self.assertTrue(self.process.terminated)
        # the next start preflights the phone again
        self.adb.failing.clear()
        self.process = _Process()
        session.start(self.root / "worker-join1.log")
        self.assertEqual(sum(row.startswith("shell sha256sum") for row in self.adb.commands), 2)

    def _lost_mid_call(self, local_client_exit, directory):
        """G1b: a started co-helper whose worker was SIGTERMed while the server's client was
        connected: its log ends at "client connected" (it never logs that the client left), no
        worker pid remains on the phone and the local adb client exits (or hangs: None)."""
        self.adb, self.process = _FakeAdb(), _Process()
        session = self._session(helper=True)
        lifecycle = CoHelperLifecycle(h.co_helpers(), {h.PIXEL: session}, IdleCoHelperStopPolicy(),
                                      identity_checks={h.PIXEL: lambda: IDENTITY})
        directory.mkdir()
        lifecycle.start_trace(directory, tolerate_absent=True)
        self.assertEqual(lifecycle.started, [h.PIXEL])
        with session._log_path.open("a") as log:
            log.write(CONNECTED_MARKER + "\n")
        self.adb.pids.clear()
        self.process.code = local_client_exit
        self.adb.commands.clear()
        return session, lifecycle

    def test_g1b_trace_end_releases_a_worker_lost_mid_call_and_removes_its_forward(self):
        """The trace-end stop read the busy log and refused ("stays running") before looking at the
        phone, so the forward stayed. Elastic trace end: the worker is gone from the phone, so it is
        released: no signal, local client ended, forward removed, receipt recorded; any exit code."""
        for code in (143, 0, None):
            with self.subTest(local_client_exit=code):
                session, lifecycle = self._lost_mid_call(code, self.root / f"exit-{code}")
                # flag absent: today's refusal, the forward is left (byte-identical behaviour)
                with self.assertRaisesRegex(PhysicalAdapterError, "stays running"):
                    lifecycle.end_trace({})
                self.assertEqual(lifecycle.started, [h.PIXEL])
                self.assertNotIn("forward --remove tcp:26991", self.adb.commands)
                (receipt,) = lifecycle.end_trace({}, continue_past_failures=True)
                self.assertEqual(
                    {key: receipt[key] for key in ("phase", "already_exited", "signalled", "forward_removed",
                                                   "boot_unchanged", "worker_pids_after", "exit_code")},
                    {"phase": "stop", "already_exited": True, "signalled": False, "forward_removed": True,
                     "boot_unchanged": True, "worker_pids_after": [], "exit_code": -15 if code is None else code})
                self.assertIn("forward --remove tcp:26991", self.adb.commands)
                self.assertEqual(self.process.terminated, code is None)
                self.assertFalse(any("kill" in row for row in self.adb.commands))
                self.assertEqual((lifecycle.started, session.active), ([], False))

    def test_elastic_trace_end_still_never_signals_a_worker_serving_a_client(self):
        session, lifecycle = self._lost_mid_call(None, self.root / "busy")
        self.adb.pids[:] = [4242]  # still on the phone, mid-call
        with self.assertRaisesRegex(PhysicalAdapterError, "stays running") as caught:
            lifecycle.end_trace({}, continue_past_failures=True)
        self.assertEqual(caught.exception.receipts, ())
        # an adb failure proves nothing about the worker: the same refusal
        self.adb.pids.clear()
        self.adb.failing.add("ps")
        with self.assertRaisesRegex(PhysicalAdapterError, "stays running"):
            session.stop(allow_idle_signal=True, release_exited=True)
        self.assertFalse(any("kill" in row for row in self.adb.commands))
        self.assertNotIn("forward --remove tcp:26991", self.adb.commands)
        self.assertTrue(session.active)
        # a reboot still fails the stop, after the forward is removed
        self.adb.failing.clear()
        self.adb.boot = "boot-2"
        self.process.code = 0
        with self.assertRaisesRegex(PhysicalAdapterError, "stop failed"):
            session.stop(allow_idle_signal=True, release_exited=True)
        self.assertIn("forward --remove tcp:26991", self.adb.commands)
        with self.assertRaisesRegex(PhysicalAdapterError, "invalid"):
            session.stop(release_exited="yes")


# ---- co-helper lifecycle and the rig membership probe --------------------------------------------

class _Receipt:
    def __init__(self, value):
        self.value = value

    def to_json(self):
        return dict(self.value)


class _ElasticWorker:
    def __init__(self, *, present=True):
        row = h.co_helpers().helpers[0]
        self.configuration = SimpleNamespace(
            device_id=row.device_id, serial=row.serial, layer_mask=row.layer_mask,
            column_quantum=row.column_quantum, max_tokens=row.max_tokens, phone_port=26990, forward_port=26991)
        self.present, self.active, self.serving, self.forward = present, False, False, dict(h.PIXEL_TRANSPORT)
        self.calls = []

    def preflight(self):
        self.calls.append("preflight")
        if not self.present:
            raise PhysicalAdapterError("error: device '" + self.configuration.serial + "' not found")
        return _Receipt({"phase": "preflight", "device_id": self.configuration.device_id})

    def start(self, log_path, *, release_failed_start=False):
        self.calls.append(("start", log_path.name))
        self.release_failed_start = release_failed_start
        self.active = self.serving = True
        return _Receipt({"phase": "launch", "device_id": self.configuration.device_id,
                         "serial": self.configuration.serial, "worker_pids": [4242]})

    def transport_parameters(self):
        return dict(self.forward)

    def alive(self):
        self.calls.append("alive")
        return self.active and self.serving

    def release_lost(self):
        self.calls.append("release_lost")
        self.active = self.serving = False
        return _Receipt({"phase": "release_lost", "device_id": self.configuration.device_id})


class _Stop:
    name = "recording"

    def __init__(self, failing=False):
        self.calls, self.failing = [], failing

    def stop(self, session, *, served_calls):
        self.calls.append(session.configuration.device_id)
        if self.failing:
            raise PhysicalAdapterError("resident adb-tcp worker stays running")
        session.active = False
        return {"phase": "stop", "device_id": session.configuration.device_id}


class _Identity:
    def __init__(self):
        self.error = None
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.error is not None:
            raise PhysicalAdapterError(self.error)
        return IDENTITY


class CoHelperMembershipLifecycleTests(unittest.TestCase):
    def _lifecycle(self, worker, stop=None, identity=None):
        return CoHelperLifecycle(h.co_helpers(), {h.PIXEL: worker}, stop or _Stop(),
                                 identity_checks=None if identity is None else {h.PIXEL: identity})

    def test_absent_helper_is_recorded_only_when_tolerated(self):
        worker = _ElasticWorker(present=False)
        with self.assertRaisesRegex(PhysicalAdapterError, "not found"):
            self._lifecycle(worker).start_trace(Path("/logs"))
        lifecycle = self._lifecycle(worker)
        receipts = lifecycle.start_trace(Path("/logs"), tolerate_absent=True)
        self.assertEqual(receipts[-1]["kind"], "DEVICE_ABSENT_AT_START")
        self.assertIn("not found", receipts[-1]["reason"])
        self.assertEqual((lifecycle.started, lifecycle.absent), ([], {h.PIXEL}))
        self.assertEqual(lifecycle.end_trace({}, continue_past_failures=True), ())

    def test_a_foreign_forward_still_fails_under_tolerance(self):
        worker = _ElasticWorker()
        worker.forward["ffn_worker_port"] = 1
        with self.assertRaisesRegex(PhysicalAdapterError, "forward differs"):
            self._lifecycle(worker).start_trace(Path("/logs"), tolerate_absent=True)

    def test_join_needs_the_pinned_identity_and_releases_a_failed_start(self):
        worker = _ElasticWorker(present=False)
        lifecycle = self._lifecycle(worker)
        lifecycle.start_trace(Path("/logs"), tolerate_absent=True)
        with self.assertRaisesRegex(PhysicalAdapterError, "pinned identity"):
            lifecycle.join(h.PIXEL, Path("/logs/join1.log"))
        identity = _Identity()
        lifecycle = self._lifecycle(worker, identity=identity)
        lifecycle.start_trace(Path("/logs"), tolerate_absent=True)
        worker.present = True
        worker.forward["ffn_worker_port"] = 1
        with self.assertRaisesRegex(PhysicalAdapterError, "forward differs"):
            lifecycle.join(h.PIXEL, Path("/logs/join1.log"))
        self.assertEqual(worker.calls[-1], "release_lost")
        self.assertEqual((lifecycle.started, lifecycle.absent), ([], {h.PIXEL}))
        worker.forward = dict(h.PIXEL_TRANSPORT)
        receipts = lifecycle.join(h.PIXEL, Path("/logs/join2.log"))
        self.assertEqual((receipts[-1]["kind"], receipts[-1]["identity_sha256"]), ("JOINED", IDENTITY))
        self.assertEqual((lifecycle.started, lifecycle.absent), ([h.PIXEL], set()))
        with self.assertRaisesRegex(PhysicalAdapterError, "not absent"):
            lifecycle.join(h.PIXEL, Path("/logs/join3.log"))

    def test_a_lost_helper_is_released_in_the_background(self):
        worker, stop = _ElasticWorker(), _Stop()
        lifecycle = self._lifecycle(worker, stop)
        lifecycle.start_trace(Path("/logs"))
        with tempfile.TemporaryDirectory() as directory:
            model = h.manifest(directory)
        thread = lifecycle.lose(h.PIXEL)
        self.assertIsInstance(thread, threading.Thread)
        self.assertEqual(lifecycle.residency_observations(model), ())
        self.assertEqual(lifecycle.wait_released(timeout_s=5), ({"phase": "release_lost", "device_id": h.PIXEL},))
        self.assertEqual(lifecycle.absent, {h.PIXEL})
        self.assertIsNone(lifecycle.lose(h.PIXEL))
        self.assertEqual(lifecycle.end_trace({}, continue_past_failures=True), ())
        self.assertEqual(stop.calls, [])

    def test_elastic_end_trace_attempts_every_helper_and_keeps_receipts(self):
        worker = _ElasticWorker()
        lifecycle = self._lifecycle(worker, _Stop(failing=True))
        lifecycle.start_trace(Path("/logs"))
        with self.assertRaisesRegex(PhysicalAdapterError, "co-helper stop failed") as caught:
            lifecycle.end_trace({}, continue_past_failures=True)
        self.assertEqual(caught.exception.receipts, ())
        with self.assertRaisesRegex(PhysicalAdapterError, "stays running"):
            lifecycle.end_trace({})


class _Scheduler:
    def __init__(self):
        self.calls, self.quarantined = [], {}

    def runtime_protected_work_end_us(self):
        return None

    def quarantine_device(self, device_id, *, reason, at_us):
        self.calls.append(("quarantine_device", device_id, reason))
        self.quarantined.setdefault(device_id, reason)

    def readmit_device(self, device_id, *, at_us, identity_sha256):
        self.calls.append(("readmit_device", device_id, identity_sha256))
        self.quarantined.pop(device_id, None)

    def quarantined_devices(self):
        return dict(self.quarantined)


class _RigMembershipCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        patcher = mock.patch.object(rig_module, "HELPER_LIVENESS_INTERVAL_S", 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _rig(self, worker, *, elastic=ELASTIC, identity=None, stop=None):
        identity = identity or _Identity()
        lifecycle = CoHelperLifecycle(h.co_helpers(), {h.PIXEL: worker}, stop or _Stop(),
                                      identity_checks={h.PIXEL: identity})
        rig = HeterogeneousPhysicalRig.__new__(HeterogeneousPhysicalRig)
        rig.configuration = SimpleNamespace(output_directory=self.root, phone_device_id=h.OP15,
                                            phone_usb_serial=h.OP15_SERIAL, adb_port=5037,
                                            minimum_usb_speed_mbps=5000)
        rig.epoch_ns = time.monotonic_ns()
        rig._lock, rig._transition_active, rig._execution_backend = threading.RLock(), False, None
        rig._scheduler = None
        rig._co_helper_lifecycles = {"sha256:" + "e" * 64: lifecycle}
        rig._co_helper_sessions = {h.PIXEL: worker}
        rig._co_helper_activity = {h.PIXEL: PhoneActivityIntervalTracker()}
        rig._co_helper_receipts = []
        rig._init_phone_membership(elastic)
        return rig, lifecycle, identity

    @staticmethod
    def _state(rig):
        return rig._membership[h.PIXEL]

    def _kinds(self, rig):
        return [row["kind"] for row in rig.helper_membership_events]


class RigMembershipTests(_RigMembershipCase):
    def test_absent_at_start_is_quarantined_once_the_scheduler_is_bound_then_joins(self):
        worker = _ElasticWorker(present=False)
        rig, lifecycle, identity = self._rig(worker)
        rig.begin_trace(time.monotonic_ns())
        self.assertEqual(lifecycle.absent, {h.PIXEL})
        row = rig._helper_membership_observations()[h.PIXEL]
        self.assertEqual((row["validity"], row["membership"], row["valid"]), ("UNAVAILABLE", "ABSENT", False))
        scheduler = _Scheduler()
        rig._bind_scheduler(scheduler)
        rig._bind_scheduler(scheduler)
        self.assertEqual(scheduler.calls, [("quarantine_device", h.PIXEL, "DEVICE_ABSENT_AT_START")])
        # still absent: the join is rejected once per reason, the device stays quarantined
        identity.error = "USB sysfs device is absent: 2-9.2"
        rig._probe_helper_membership(h.PIXEL)
        self._state(rig).last_join_ns = 0
        rig._probe_helper_membership(h.PIXEL)
        self.assertEqual(self._kinds(rig).count("JOIN_REJECTED"), 1)
        # plugged in: identity-pinned join, readmission, the telemetry row goes
        identity.error, worker.present = None, True
        self._state(rig).last_join_ns = 0
        value = rig._probe_helper_membership(h.PIXEL)
        self.assertEqual((value["state"], value["readmissions"], value["identity_sha256"]), ("MEMBER", 1, IDENTITY))
        self.assertEqual(scheduler.calls[-1], ("readmit_device", h.PIXEL, IDENTITY))
        self.assertEqual(rig._helper_membership_observations(), {})
        self.assertIn(("start", "PIXEL0-worker-join3.log"), worker.calls)
        saved = json.loads((self.root / "HELPER_MEMBERSHIP.json").read_text())
        self.assertEqual([row["kind"] for row in saved], ["DEVICE_ABSENT_AT_START", "JOIN_REJECTED", "JOINED"])

    def test_liveness_loss_quarantines_releases_and_rejoins_after_the_cooldown(self):
        worker = _ElasticWorker()
        elastic = elastic_phones_configuration({"join": True, "join_probe_interval_s": 1,
                                                "readmission_cooldown_s": 60, "max_readmissions_per_device": 1})
        rig, lifecycle, _identity = self._rig(worker, elastic=elastic)
        rig.begin_trace(time.monotonic_ns())
        scheduler = _Scheduler()
        rig._bind_scheduler(scheduler)
        self.assertEqual(scheduler.calls, [])
        worker.serving = False
        rig._probe_helper_membership(h.PIXEL)
        self.assertEqual(self._state(rig).state, "MEMBER")
        rig._probe_helper_membership(h.PIXEL)
        self.assertEqual(self._state(rig).state, "QUARANTINED")
        self.assertEqual(scheduler.calls, [("quarantine_device", h.PIXEL, "HELPER_LOST")])
        lifecycle.wait_released(timeout_s=5)
        self.assertIn("release_lost", worker.calls)
        # the cooldown holds the join back
        worker.calls.clear()
        rig._probe_helper_membership(h.PIXEL)
        self.assertNotIn("preflight", worker.calls)
        self._state(rig).since_ns -= 61 * 10**9
        rig._probe_helper_membership(h.PIXEL)
        self.assertEqual(self._state(rig).state, "MEMBER")
        self.assertEqual(scheduler.calls[-1][0], "readmit_device")
        # a second loss: the readmission limit keeps it out for the rest of the run
        worker.serving = False
        rig._probe_helper_membership(h.PIXEL)
        rig._probe_helper_membership(h.PIXEL)
        lifecycle.wait_released(timeout_s=5)
        self._state(rig).since_ns -= 61 * 10**9
        rig._probe_helper_membership(h.PIXEL)
        rig._probe_helper_membership(h.PIXEL)
        self.assertEqual(self._state(rig).state, "QUARANTINED")
        self.assertEqual(self._kinds(rig).count("READMISSION_LIMIT"), 1)
        self.assertEqual([call[0] for call in scheduler.calls],
                         ["quarantine_device", "readmit_device", "quarantine_device"])

    def test_g1b_an_exited_worker_client_is_lost_at_the_first_check(self):
        """Two failed checks one interval apart (one adb hiccup never quarantines a phone) put a
        dead worker's loss 5-11 s after its exit, past G1b's fail-fast end of the trace. A worker
        whose own adb client exited cannot come back, so it is lost at the first check (journaled);
        an adb failure still needs the second check, and a failing exit probe changes nothing."""
        worker = _ElasticWorker()
        rig, lifecycle, _identity = self._rig(worker)
        rig.begin_trace(time.monotonic_ns())
        scheduler = _Scheduler()
        rig._bind_scheduler(scheduler)
        worker.serving, worker.client_exited = False, lambda: False
        rig._probe_helper_membership(h.PIXEL)
        self.assertEqual(self._state(rig).state, "MEMBER")
        worker.serving = True
        rig._probe_helper_membership(h.PIXEL)

        def unreadable():
            raise OSError("adb gone")

        worker.serving, worker.client_exited = False, unreadable
        rig._probe_helper_membership(h.PIXEL)
        self.assertEqual(self._state(rig).state, "MEMBER")
        worker.serving = True
        rig._probe_helper_membership(h.PIXEL)
        self.assertEqual(scheduler.calls, [])
        # the worker died: its adb client exited
        worker.serving, worker.client_exited = False, lambda: True
        rig._probe_helper_membership(h.PIXEL)
        self.assertEqual(self._state(rig).state, "QUARANTINED")
        self.assertEqual(scheduler.calls, [("quarantine_device", h.PIXEL, "HELPER_LOST")])
        rows = [json.loads(line) for line in
                (self.root / "HELPER_MEMBERSHIP_PROBE.jsonl").read_text().splitlines()]
        self.assertEqual([(row["alive"], row["client_exited"], row["liveness_failures"]) for row in rows],
                         [(False, False, 1), (True, False, 0), (False, False, 1), (True, False, 0),
                          (False, True, 1)])
        lifecycle.wait_released(timeout_s=5)
        self.assertIn("release_lost", worker.calls)

    def test_identity_mismatch_keeps_the_device_quarantined(self):
        worker = _ElasticWorker()
        rig, lifecycle, identity = self._rig(worker)
        rig.begin_trace(time.monotonic_ns())
        scheduler = _Scheduler()
        rig._bind_scheduler(scheduler)
        # the execution path reported the loss: the rig releases the worker without re-reporting it
        scheduler.quarantined[h.PIXEL] = "HELPER_LOST"
        rig._probe_helper_membership(h.PIXEL)
        self.assertEqual(self._state(rig).state, "QUARANTINED")
        self.assertEqual(scheduler.calls, [])
        lifecycle.wait_released(timeout_s=5)
        identity.error = "helper kernel differs from qualification"
        rig._probe_helper_membership(h.PIXEL)
        self.assertEqual(self._state(rig).state, "QUARANTINED")
        rejected = [row for row in rig.helper_membership_events if row["kind"] == "JOIN_REJECTED"]
        self.assertIn("kernel differs", rejected[0]["reason"])
        self.assertNotIn("readmit_device", [call[0] for call in scheduler.calls])
        self.assertEqual(scheduler.quarantined, {h.PIXEL: "HELPER_LOST"})

    def test_trace_end_closes_membership_and_stops_every_helper(self):
        worker, stop = _ElasticWorker(), _Stop()
        rig, lifecycle, _identity = self._rig(worker, stop=stop)
        rig.begin_trace(time.monotonic_ns())
        rig._stop_co_helpers()
        self.assertEqual(stop.calls, [h.PIXEL])
        self.assertEqual(rig._probe_helper_membership(h.PIXEL)["state"], "CLOSED")
        receipts = json.loads((self.root / "CO_HELPER_LIFECYCLE.json").read_text())
        self.assertEqual(receipts[-1], {"phase": "stop", "device_id": h.PIXEL})

    def test_primary_phone_rejoins_through_adb_and_its_usb_identity(self):
        worker = _ElasticWorker()
        rig, _lifecycle, _identity = self._rig(worker)
        rig.begin_trace(time.monotonic_ns())
        scheduler = _Scheduler()
        rig._bind_scheduler(scheduler)
        self.assertEqual(rig._probe_primary_membership()["state"], "MEMBER")
        scheduler.quarantined[h.OP15] = "HELPER_LOST"
        with mock.patch.object(rig_module, "verify_android_usb_restored",
                               side_effect=PhysicalAdapterError("ADB is unavailable")):
            self.assertEqual(rig._probe_primary_membership()["state"], "QUARANTINED")
        restored = SimpleNamespace(serial=h.OP15_SERIAL, vendor_id="18d1", product_id="4ee7")
        rig._membership[h.OP15].last_join_ns = 0
        with mock.patch.object(rig_module, "verify_android_usb_restored", return_value=restored):
            self.assertEqual(rig._probe_primary_membership()["state"], "MEMBER")
        self.assertEqual(scheduler.calls[-1][:2], ("readmit_device", h.OP15))
        self.assertEqual(scheduler.quarantined, {})


class _SlowScheduler(_Scheduler):
    """Holds its first quarantine until released (a binding flush in progress)."""

    def __init__(self):
        super().__init__()
        self.entered, self.release = threading.Event(), threading.Event()

    def quarantine_device(self, device_id, *, reason, at_us):
        self.entered.set()
        self.release.wait(5)
        super().quarantine_device(device_id, reason=reason, at_us=at_us)


class _FailingOnceScheduler(_Scheduler):
    def __init__(self):
        super().__init__()
        self.failures = 1

    def quarantine_device(self, device_id, *, reason, at_us):
        if self.failures:
            self.failures -= 1
            self.calls.append(("quarantine_failed", device_id, reason))
            raise RuntimeError("scheduler is not ready")
        super().quarantine_device(device_id, reason=reason, at_us=at_us)


class _KernelSession:
    """``AdbTcpPhoneWorkerSession`` stand-in for HelperPhoneEvidence.verify_identity's ``uname -r``."""

    shell = None

    def __init__(self, _worker):
        pass

    def _shell(self, command):
        return type(self).shell(command)


def helper_evidence():
    from research_dev.scheduler.campaigns.burstgpt.helper_phone_evidence import HelperPhoneEvidence

    identity = SimpleNamespace(
        hardware_identity={"phone_usb_sysfs_device": "2-9.2", "host_usb_controller": "0000:00:14.0",
                           "adb_usb_identity": "18d1:4ee7", "phone_kernel_release": "6.6.30-android15"},
        minimum_usb_speed_mbps=5000, identity_sha256=IDENTITY)
    return HelperPhoneEvidence(path=Path("/evidence.json"), identity=identity,
                               worker=SimpleNamespace(serial=h.PIXEL_SERIAL), profile_fragment={}, power=None)


def usb_port(serial=h.PIXEL_SERIAL):
    return SimpleNamespace(serial=serial, negotiated_speed_mbps=5000, host_controller="0000:00:14.0",
                           vendor_product="18d1:4ee7")


class MembershipReviewFixTests(_RigMembershipCase):
    """Adversarial-review findings M1, M3, m3, m4 (flagged rigs only)."""

    def test_m1_the_real_identity_check_rejects_the_join_with_its_reason(self):
        from research_dev.scheduler.campaigns.burstgpt import helper_phone_evidence as evidence_module

        evidence = helper_evidence()
        worker = _ElasticWorker(present=False)
        rig, lifecycle, _identity = self._rig(worker, identity=evidence.verify_identity)
        rig.begin_trace(time.monotonic_ns())
        scheduler = _Scheduler()
        rig._bind_scheduler(scheduler)
        worker.present = True
        # a different phone on the qualified port: the real check's own refusal
        with mock.patch.object(evidence_module, "observe_usb_port", return_value=usb_port("OTHER")):
            rig._probe_helper_membership(h.PIXEL)
        # an error of another type (adb answers with undecodable bytes): still a recorded rejection,
        # not an exception escaping the probe thread (the join would then be retried silently)
        _KernelSession.shell = mock.Mock(side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid"))
        self._state(rig).last_join_ns = 0
        with mock.patch.object(evidence_module, "observe_usb_port", return_value=usb_port()), \
                mock.patch.object(evidence_module, "AdbTcpPhoneWorkerSession", _KernelSession):
            value = rig._probe_helper_membership(h.PIXEL)
        self.assertEqual(value["state"], "ABSENT")
        rejected = [row["reason"] for row in rig.helper_membership_events if row["kind"] == "JOIN_REJECTED"]
        self.assertEqual(len(rejected), 2)
        self.assertIn("PhysicalAdapterError: helper USB identity differs from qualification", rejected[0])
        self.assertTrue(rejected[1].startswith("UnicodeDecodeError"))
        self.assertEqual((lifecycle.absent, lifecycle.started), ({h.PIXEL}, []))
        self.assertNotIn("readmit_device", [call[0] for call in scheduler.calls])
        # the pinned phone: the real check passes and the helper joins
        _KernelSession.shell = mock.Mock(return_value="6.6.30-android15\n")
        self._state(rig).last_join_ns = 0
        with mock.patch.object(evidence_module, "observe_usb_port", return_value=usb_port()), \
                mock.patch.object(evidence_module, "AdbTcpPhoneWorkerSession", _KernelSession):
            value = rig._probe_helper_membership(h.PIXEL)
        self.assertEqual((value["state"], value["identity_sha256"]), ("MEMBER", IDENTITY))
        self.assertEqual(scheduler.calls[-1], ("readmit_device", h.PIXEL, IDENTITY))

    def test_m3_a_scheduler_quarantine_keeps_new_requests_off_the_device_at_once(self):
        with h.TemporaryModel() as model:
            catalog = h.runtime_catalog(model, declaration=h.co_helpers())
            scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
            scheduler.register_runtime_capabilities(catalog)
            scheduler.register_model_manifest(model)
            worker = _ElasticWorker()
            rig, _lifecycle, _identity = self._rig(worker)
            rig.begin_trace(time.monotonic_ns())
            rig._bind_scheduler(scheduler)

            def split_reasons(request_id):
                value = h.snapshot(model, catalog)
                value = replace(value, telemetry_observations={
                    **value.telemetry_observations, **rig._helper_membership_observations()})
                values = scheduler.generate_automated_candidates(h.request(request_id), model.model_id, value)
                return next(row for row in values.candidates
                            if row.binding.executor_id.endswith(":operator_split")).rejection_reasons

            self.assertNotIn("CO_HELPER_UNAVAILABLE", split_reasons("before"))
            # the execution path's helper_lost quarantine; the probe thread has not run
            scheduler.quarantine_device(h.PIXEL, reason="HELPER_LOST", at_us=5)
            self.assertEqual(self._state(rig).state, "MEMBER")
            row = rig._helper_membership_observations()[h.PIXEL]
            self.assertEqual((row["membership"], row["membership_reason"]), ("QUARANTINED", "HELPER_LOST"))
            self.assertIn("CO_HELPER_UNAVAILABLE", split_reasons("after"))
            scheduler.readmit_device(h.PIXEL, at_us=6, identity_sha256=IDENTITY)
            self.assertEqual(rig._helper_membership_observations(), {})

    def test_m3_a_probe_readmission_never_overtakes_the_quarantine_queued_before_the_binding(self):
        worker = _ElasticWorker(present=False)
        rig, _lifecycle, _identity = self._rig(worker)
        rig.begin_trace(time.monotonic_ns())
        worker.present = True
        scheduler = _SlowScheduler()
        binder = threading.Thread(target=rig._bind_scheduler, args=(scheduler,))
        binder.start()
        try:
            self.assertTrue(scheduler.entered.wait(5))
            # the probe joins while the queued ABSENT quarantine is being delivered
            self.assertEqual(rig._probe_helper_membership(h.PIXEL)["state"], "MEMBER")
        finally:
            scheduler.release.set()
            binder.join(5)
        self.assertEqual([call[0] for call in scheduler.calls], ["quarantine_device", "readmit_device"])
        self.assertEqual(scheduler.quarantined, {})
        self.assertIs(rig._membership_scheduler, scheduler)

    def test_m3_a_failed_flush_keeps_every_queued_call_for_the_next_binding(self):
        worker = _ElasticWorker(present=False)
        rig, _lifecycle, _identity = self._rig(worker)
        rig.begin_trace(time.monotonic_ns())
        worker.present = True
        self.assertEqual(rig._probe_helper_membership(h.PIXEL)["state"], "MEMBER")
        self.assertEqual([row[0] for row in rig._membership_pending], ["quarantine_device", "readmit_device"])
        scheduler = _FailingOnceScheduler()
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            rig._bind_scheduler(scheduler)
        self.assertEqual([row[0] for row in rig._membership_pending], ["quarantine_device", "readmit_device"])
        self.assertIsNone(rig._membership_scheduler)
        self.assertIn("SCHEDULER_MEMBERSHIP_FAILED", self._kinds(rig))
        rig._bind_scheduler(scheduler)
        self.assertEqual([call[0] for call in scheduler.calls],
                         ["quarantine_failed", "quarantine_device", "readmit_device"])
        self.assertEqual((scheduler.quarantined, rig._membership_pending), ({}, []))

    def _primary_rig(self, elastic=ELASTIC):
        rig, _lifecycle, _identity = self._rig(_ElasticWorker(), elastic=elastic)
        rig.begin_trace(time.monotonic_ns())
        scheduler = _Scheduler()
        rig._bind_scheduler(scheduler)
        scheduler.quarantined[h.OP15] = "HELPER_LOST"
        return rig, scheduler

    def test_m4_an_adb_timeout_is_a_recorded_rejection(self):
        rig, scheduler = self._primary_rig()
        with mock.patch.object(rig_module, "verify_android_usb_restored",
                               side_effect=subprocess.TimeoutExpired(["adb", "get-state"], 5)):
            self.assertEqual(rig._probe_primary_membership()["state"], "QUARANTINED")
        (rejected,) = [row for row in rig.helper_membership_events if row["kind"] == "JOIN_REJECTED"]
        self.assertEqual((rejected["device_id"], rejected["reason"][:14]), (h.OP15, "TimeoutExpired"))
        self.assertEqual(scheduler.quarantined, {h.OP15: "HELPER_LOST"})

    def test_m4_no_primary_readmission_after_the_trace_closed(self):
        rig, scheduler = self._primary_rig()
        restored = SimpleNamespace(serial=h.OP15_SERIAL, vendor_id="18d1", product_id="4ee7")

        def close_during_the_check(**_arguments):
            with rig._membership_lock:
                rig._membership_closed = True
            return restored

        with mock.patch.object(rig_module, "verify_android_usb_restored", side_effect=close_during_the_check):
            self.assertEqual(rig._probe_primary_membership()["state"], "CLOSED")
        self.assertNotIn("readmit_device", [call[0] for call in scheduler.calls])
        self.assertNotIn("JOINED", self._kinds(rig))
        self.assertEqual(rig._membership[h.OP15].state, "QUARANTINED")

    def test_m4_primary_probe_is_exclusive_and_the_trace_end_waits_for_it(self):
        rig, _scheduler = self._primary_rig()
        rig._primary_membership_busy.acquire()
        closer = threading.Thread(target=rig._close_phone_membership)
        try:
            self.assertEqual(rig._probe_primary_membership()["state"], "BUSY")
            closer.start()
            closer.join(0.2)
            self.assertTrue(closer.is_alive())
        finally:
            rig._primary_membership_busy.release()
        closer.join(5)
        self.assertFalse(closer.is_alive())
        self.assertEqual(rig._probe_primary_membership()["state"], "CLOSED")

    def test_m4_primary_rejoins_are_capped(self):
        elastic = elastic_phones_configuration({"join": True, "join_probe_interval_s": 1,
                                                "readmission_cooldown_s": 0, "max_readmissions_per_device": 1})
        rig, scheduler = self._primary_rig(elastic)
        restored = SimpleNamespace(serial=h.OP15_SERIAL, vendor_id="18d1", product_id="4ee7")
        with mock.patch.object(rig_module, "verify_android_usb_restored", return_value=restored) as verify:
            self.assertEqual(rig._probe_primary_membership()["state"], "MEMBER")
            scheduler.quarantined[h.OP15] = "HELPER_LOST"
            rig._membership[h.OP15].last_join_ns = 0
            self.assertEqual(rig._probe_primary_membership()["state"], "QUARANTINED")
            rig._membership[h.OP15].last_join_ns = 0
            self.assertEqual(rig._probe_primary_membership()["state"], "QUARANTINED")
        self.assertEqual(verify.call_count, 1)
        self.assertEqual(self._kinds(rig).count("READMISSION_LIMIT"), 1)
        self.assertEqual([call[0] for call in scheduler.calls], ["readmit_device"])


def _sse(tokens, predicted, *, stop=False):
    body = {"content": "x" * len(tokens), "id_slot": 0, "stop": stop, "tokens": tokens,
            "tokens_predicted": predicted}
    return ("data: " + json.dumps(body, sort_keys=True) + "\n\n").encode("ascii")


class HelperLossEndToEndTests(unittest.TestCase):
    """T1 recorded end to end: an adaptive-split ticket (OP15 + Pixel co-helper) streams tokens and
    decode boundaries, the llama.cpp stream then breaks (HTTP backend) with the server's
    ``S41SERVERFFNERROR helper=pixel`` line; the rig's classifier names the Pixel, the adapter calls the
    scheduler's REAL ``quarantine_device`` and the FALLBACK re-executes the request from its prompt on
    the desktop; the rig then keeps new requests off the Pixel and releases its worker."""

    def setUp(self):
        scope = h.TemporaryModel()
        self.model = scope.__enter__()
        self.addCleanup(scope.__exit__)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.catalog = h.runtime_catalog(self.model, declaration=h.co_helpers())
        self.scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        self.scheduler.register_runtime_capabilities(self.catalog)
        self.scheduler.register_model_manifest(self.model)
        self.snapshot = h.snapshot(self.model, self.catalog)

    def _rig(self, environment):
        worker = _ElasticWorker()
        rig = HeterogeneousPhysicalRig.__new__(HeterogeneousPhysicalRig)
        rig.configuration = SimpleNamespace(output_directory=self.root, phone_device_id=h.OP15,
                                            resident_executor_id="physical:resident", elastic_phones=ELASTIC)
        rig.epoch_ns = time.monotonic_ns()
        rig._lock, rig._transition_active, rig._scheduler = threading.RLock(), False, None
        rig._live_executors, rig._execution_markers = {}, {}
        lifecycle = CoHelperLifecycle(h.co_helpers(), {h.PIXEL: worker}, _Stop(),
                                      identity_checks={h.PIXEL: _Identity()})
        rig._co_helper_lifecycles = {self.model.artifact_sha256: lifecycle}
        rig._co_helper_sessions = {h.PIXEL: worker}
        rig._co_helper_activity = {h.PIXEL: PhoneActivityIntervalTracker()}
        rig._co_helper_receipts = []
        rig._init_phone_membership(ELASTIC)
        rig._server = SimpleNamespace(
            process=SimpleNamespace(pid=4242), environment=dict(environment), exit_code=lambda: None,
            failure_evidence=lambda _index, *, timeout_s, decisive=None: ((
                "srv  update_slots: Compute aborted. off = 0, n_batch = 1, ret = 2",
                "S41SERVERFFNERROR helper=pixel detail=recv failed: connection reset",
            ), None))
        return rig, lifecycle, worker

    def test_stream_failure_quarantines_the_pixel_and_recovers_on_the_desktop(self):
        from research_dev.scheduler import AdaptiveDecodeConfig
        from research_dev.scheduler._internal.adaptive_decode_contracts import AdaptiveDecodeRawWindowObservation
        from research_dev.scheduler.adapters import CanonicalPhysicalAdapter, interpret_runtime_ticket
        from research_dev.scheduler.adapters.contracts import (
            CompletionStreamError, RawEnergyMeasurement, RawExecutionObservation, RawTransitionObservation,
        )
        from research_dev.scheduler.adapters.http_backend import (
            CanonicalHttpExecutionBackend, LlamaCppCompletionPayload, LlamaCppHttpClient,
        )
        from research_dev.scheduler.adapters.llama_server import llama_server_launch_contract

        scheduler, snapshot, catalog = self.scheduler, self.snapshot, self.catalog
        ticket = scheduler.submit_automated_request(h.request("t1", output_tokens=30), self.model.model_id,
                                                    snapshot, selection_mode="adaptive-decode")
        self.assertEqual(ticket.execution_plan.execution_contract.execution_mode, "adaptive-split")
        self.assertIn(h.PIXEL, ticket.execution_plan.device_ids)
        environment = llama_server_launch_contract(interpret_runtime_ticket(ticket), self.model).ffn_environment
        rig, lifecycle, worker = self._rig(environment)
        rig.begin_trace(rig.epoch_ns)
        epoch_ns = time.monotonic_ns() - ticket.decision.start_us * 1_000
        now_us = lambda: (time.monotonic_ns() - epoch_ns) // 1_000  # noqa: E731
        streamed, remaining_at_fallback = {}, []

        class StreamingClient(LlamaCppHttpClient):
            """llama.cpp SSE stream of the helper-assisted attempt: tokens, two decode boundaries, then
            the server's error chunk once the Pixel's FFN client failed."""

            def complete(self, endpoint, payload, control_check, **_options):
                with payload.stream_path.open("xb") as stream:
                    for index in range(3):
                        stream.write(_sse([100 + index], index + 1))
                payload.on_first_token(time.monotonic_ns())
                directive = scheduler.start_adaptive_decode(
                    "t1", slot_id=0, first_token_index=1, at_us=now_us(), config=AdaptiveDecodeConfig(
                        minimum_remaining_tokens=4, minimum_window_tokens=2, maximum_window_tokens=2,
                        maximum_probe_tokens=20, maximum_probe_candidates=2, measurement_resolution_us=1,
                        transition_cost_us=1, transition_energy_uj=1, warmup_windows_per_policy=0))
                boundary = scheduler.adaptive_decode_boundary(
                    "t1", slot_id=0, token_index=directive.target_token_index, at_us=now_us()).boundary
                scheduler.record_adaptive_decode_window("t1", boundary, AdaptiveDecodeRawWindowObservation(
                    fleet_energy_uj_by_domain={"fleet": 200}, phone_compute_us=0, usb_transfer_us=0, rpc_us=0,
                    exposed_tail_us=0, output_valid=True, evidence_ids=("recorded-window",),
                    energy_boundary_id=catalog.placement_profile.energy_boundary_id,
                    energy_attribution_kind="isolated"))
                streamed["boundary_token"] = directive.target_token_index
                streamed["remaining"] = scheduler.record_runtime_decode_progress("t1", token_index=3, at_us=now_us())
                raise CompletionStreamError("completion stream chunk is invalid",
                                            server_error_message="Compute aborted.")

        def begin(command):
            rig._execution_markers[command.ticket_id] = (rig._server, None, SimpleNamespace(stderr_index=0), (8, 30))

        http = CanonicalHttpExecutionBackend(
            StreamingClient(), SimpleNamespace(measure=lambda *_args: None), epoch_ns=epoch_ns,
            on_execution_start=begin,
            on_execution_finish=lambda command: rig._execution_markers.pop(command.ticket_id, None),
            failure_classifier=rig._classify_execution_failure)

        class Backend:
            """The helper-assisted attempt runs through the HTTP backend; the recovery is measured."""

            commands = []

            def bind_scheduler(self, bound):
                http.bind_scheduler(bound)

            def apply_transition(self, command, payload, control_check):
                control_check()
                return RawTransitionObservation(started_us=now_us(), finished_us=now_us() + 1, status="COMPLETED",
                                                evicted_artifact_sha256s=())

            def execute(self, command, payload, control_check):
                self.commands.append(command)
                if len(self.commands) == 1:
                    return http.execute(command, payload, control_check)
                control_check()
                remaining_at_fallback.append(
                    scheduler._model_placement_controller.remaining_request_decode_tokens("t1", 30))
                with payload.stream_path.open("xb") as stream:
                    for index in range(payload.output_tokens):
                        stream.write(_sse([200 + index], index + 1))
                    stream.write(_sse([], payload.output_tokens, stop=True))
                domains = sorted("energy:" + device for device in catalog.placement_profile.devices)
                return RawExecutionObservation(
                    started_us=command.planned_start_us, finished_us=command.planned_finish_us,
                    output_sha256="f" * 64, payload={"tokens": [200]},
                    energy=RawEnergyMeasurement(
                        energy_boundary_id=catalog.placement_profile.energy_boundary_id,
                        fleet_energy_uj_by_domain={domain: 100 for domain in domains},
                        transfer_energy_uj_by_link={
                            row.removeprefix("link:"): 1 for row in command.operator_plan["resource_ids"]
                            if row.startswith("link:")},
                        measurement_evidence_ids=("recorded",), attribution_kind="matched_abba"))

        backend = Backend()
        payload = LlamaCppCompletionPayload(
            request_id="t1", expected_model_alias="two-phone", input_tokens=8, output_tokens=30,
            prompt_tokens=tuple(range(8)), seed=0, stream_path=self.root / "request-000.raw",
            on_first_token=lambda _ns: None)
        adapter = CanonicalPhysicalAdapter(scheduler, backend, epoch_ns=epoch_ns,
                                           snapshot_provider=lambda _ticket, _at_us: snapshot,
                                           lease_guard_us=50_000, lease_quantum_us=100_000)
        result = adapter.execute(ticket, payload)

        # the failed attempt had streamed tokens and a decode boundary through the Pixel's split
        self.assertEqual((streamed["boundary_token"], streamed["remaining"]), (3, 27))
        self.assertIn(h.PIXEL, backend.commands[0].operator_plan["device_ids"])
        # FALLBACK: from the prompt, on the desktop, without the lost device; progress starts over
        self.assertEqual(result.ticket.previous_ticket_id, ticket.ticket_id)
        self.assertEqual(result.ticket.execution_plan.execution_contract.execution_mode, "desktop")
        self.assertNotIn(h.PIXEL, result.ticket.execution_plan.device_ids)
        self.assertEqual(remaining_at_fallback, [30])
        records = [row["event_kind"] for row in scheduler.runtime_decision_log()["records"]
                   if row["request_ids"] == ["t1"]]
        self.assertEqual(records, ["DECISION", "ACQUIRED", "FALLBACK", "ACQUIRED", "COMPLETED"])
        grouped = scheduler.adaptive_decode_grouped_observation("t1")
        self.assertEqual(grouped.terminal_status, "FAILED")
        (event,) = result.recovery_events
        self.assertEqual((event["failure_kind"], event["device_id"], event["quarantined_device_ids"],
                          event["tokens_discarded"], event["attempt_stream"]),
                         ("helper_lost", h.PIXEL, [h.PIXEL], 3, "request-000.raw.attempt1"))
        # the REAL scheduler quarantine
        self.assertEqual(scheduler.quarantined_devices(), {h.PIXEL: "HELPER_LOST"})
        self.assertEqual([(row["kind"], row["device_id"]) for row in scheduler.device_membership_events()],
                         [("DEVICE_QUARANTINED", h.PIXEL)])
        self.assertIn(b"\"stop\": true", payload.stream_path.read_bytes())
        # the run continues: new requests stay off the Pixel before its membership probe ran ...
        rig._bind_scheduler(scheduler)
        value = replace(snapshot, telemetry_observations={
            **snapshot.telemetry_observations, **rig._helper_membership_observations()})
        split = next(row for row in scheduler.generate_automated_candidates(
            h.request("t2"), self.model.model_id, value).candidates
            if row.binding.executor_id.endswith(":operator_split"))
        self.assertIn("CO_HELPER_UNAVAILABLE", split.rejection_reasons)
        # ... and the probe reconciles the loss: the worker is released, the join path takes over
        self.assertEqual(rig._probe_helper_membership(h.PIXEL)["state"], "QUARANTINED")
        lifecycle.wait_released(timeout_s=5)
        self.assertIn("release_lost", worker.calls)
        self.assertEqual(lifecycle.absent, {h.PIXEL})

    def test_g1b_boundary_stats_failure_on_an_exited_server_recovers_through_a_reload(self):
        """G1b recorded: the two-phone adaptive route's server is resident (no load), the request
        streams through the REAL adaptive controller (control apply, boundary); the Pixel's worker
        dies, the server releases the slot, so the boundary's stats call (real HTTP client) raises
        StalePhysicalSlotError, the window discard refuses it, and the server exits after printing
        ``S41SERVERFFNERROR helper=pixel``. Expect helper_lost AND server_exited semantics together:
        the classifier waits for the exit, reaps the server (one SERVER_EXITED), the FALLBACK runs on
        the desktop through a load, REQUEST_RECOVERED, DEVICE_QUARANTINED for the Pixel, no abort."""
        import http.server

        from research_dev.scheduler.adapters import CanonicalPhysicalAdapter, interpret_runtime_ticket
        from research_dev.scheduler.adapters.contracts import (
            RawEnergyMeasurement, RawExecutionObservation, RawTransitionObservation,
        )
        from research_dev.scheduler.adapters.heterogeneous_rig import _LiveExecutorResidency
        from research_dev.scheduler.adapters.http_backend import (
            CanonicalHttpExecutionBackend, LlamaCppCompletionPayload, LlamaCppHttpClient,
        )
        from research_dev.scheduler.adapters.llama_server import llama_server_launch_contract

        split = "physical:two:phone-assisted:operator_split"
        scheduler, catalog = self.scheduler, self.catalog
        hot = h.with_residency_executor(h.snapshot(self.model, catalog, desktop_hot=True), split)
        ticket = scheduler.submit_automated_request(h.request("g1b", output_tokens=30), self.model.model_id,
                                                    hot, selection_mode="adaptive-decode")
        self.assertEqual((ticket.binding.executor_id, ticket.execution_plan.transitions), (split, ()))
        self.assertIn(h.PIXEL, ticket.execution_plan.device_ids)
        command = interpret_runtime_ticket(ticket)
        rig, lifecycle, _worker = self._rig(llama_server_launch_contract(command, self.model).ffn_environment)
        rig._failure_evidence_timeout_s = 0.5
        rig._dormant_forget_server = mock.Mock()

        class DyingServer:
            """Prints the helper loss, then exits only after the failed request's slot is gone."""

            def __init__(self, environment):
                self.process, self.environment = SimpleNamespace(pid=4242), dict(environment)
                self.returncode, self.slot_released, self.stop, self.waits = None, False, mock.Mock(), []

            def exit_code(self):
                return self.returncode

            def failure_evidence(self, index, *, timeout_s, decisive=None):
                self.waits.append(timeout_s)
                if self.slot_released and timeout_s > 0:
                    self.returncode = 1  # server-context.cpp throws after send_error
                return ((
                    "S41SERVERFFNERROR helper=pixel detail=FFN split EXECUTE header exchange failed",
                    "srv  update_slots: decode() failed: Compute aborted.",
                    "srv    send_error: task id = 7, error: Compute aborted.",
                )[index:], self.returncode)

        server = DyingServer(rig._server.environment)
        rig._live_executors[split] = _LiveExecutorResidency(
            executor_id=split, endpoint=command.endpoint, server=server, manifest=self.model, parameters={},
            operator_plan={}, generation=3, participant_device_ids=ticket.execution_plan.device_ids,
            replacement_resource_ids=(), session_resource_ids=())
        rig.begin_trace(rig.epoch_ns)

        class Released(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - http.server API
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                body = json.dumps({"success": False, "message": "request and active slot differ"}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                return

        control = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Released)
        threading.Thread(target=control.serve_forever, daemon=True).start()
        self.addCleanup(control.server_close)
        self.addCleanup(control.shutdown)
        epoch_ns = time.monotonic_ns() - ticket.decision.start_us * 1_000
        now_us = lambda: (time.monotonic_ns() - epoch_ns) // 1_000  # noqa: E731
        domains = sorted("energy:" + device for device in catalog.placement_profile.devices)

        def energy(links=()):
            return RawEnergyMeasurement(
                energy_boundary_id=catalog.placement_profile.energy_boundary_id,
                fleet_energy_uj_by_domain={domain: 100 for domain in domains},
                transfer_energy_uj_by_link={link: 1 for link in links},
                measurement_evidence_ids=("recorded",), attribution_kind="matched_abba")

        calls, counters = [], dict.fromkeys((
            "calls", "desktop_compute_us", "download_bytes", "exposed_tail_us", "phone_compute_us",
            "rpc_us", "upload_bytes", "usb_transfer_us", "useful_overlap_us"), 0)

        class StreamingClient(LlamaCppHttpClient):
            """llama.cpp SSE of the two-phone attempt, token by token through the payload's decode
            progress (the real adaptive controller); the server acknowledges the policy it applies."""

            def __init__(self):
                super().__init__(slots_probe=lambda *_args: [{"id": 0, "id_task": 7, "is_processing": True}])

            def complete(self, endpoint, payload, control_check, **_options):
                with payload.stream_path.open("xb") as stream:
                    for index in range(payload.output_tokens):
                        stream.write(_sse([100 + index], index + 1))
                        if index == 0:
                            payload.on_first_token(time.monotonic_ns())
                        calls.append(("token", index + 1))
                        payload.on_decode_progress(0, index + 1, time.monotonic_ns(), False)
                raise AssertionError("the decode boundary did not fail")

            def apply_ffn_control(self, endpoint, control_value, *, timeout_s=5):
                calls.append(("control", control_value.policy.baseline))
                counters["calls"] += 1
                return ({"success": True, "slot_id": 0, "plan_generation": control_value.plan_generation,
                         "applied_token_index": calls[-2][1], "policy_hash": control_value.policy.policy_hash,
                         "runtime_stats": LlamaCppHttpClient._runtime_stats(dict(counters))},
                        time.monotonic_ns())

            def read_ffn_stats(self, endpoint, request_id, slot_id, *, timeout_s=5):
                calls.append(("stats", request_id))
                server.slot_released = True
                return LlamaCppHttpClient.read_ffn_stats(
                    f"http://127.0.0.1:{control.server_address[1]}", request_id, slot_id, timeout_s=timeout_s)

        def begin(started):
            rig._execution_markers[started.ticket_id] = (server, None, SimpleNamespace(stderr_index=0), (8, 30))

        http_backend = CanonicalHttpExecutionBackend(
            StreamingClient(), SimpleNamespace(measure=lambda *_args: energy()), epoch_ns=epoch_ns,
            on_execution_start=begin,
            on_execution_finish=lambda finished: rig._execution_markers.pop(finished.ticket_id),
            failure_classifier=rig._classify_execution_failure)

        class Backend:
            commands, transitions = [], []

            def bind_scheduler(self, bound):
                http_backend.bind_scheduler(bound)

            def apply_transition(self, transition, payload, control_check):
                control_check()
                self.transitions.append(transition.transition.transition_id)
                return RawTransitionObservation(started_us=now_us(), finished_us=now_us() + 1, status="COMPLETED",
                                                evicted_artifact_sha256s=())

            def execute(self, executed, payload, control_check):
                self.commands.append(executed)
                if len(self.commands) == 1:
                    return http_backend.execute(executed, payload, control_check)
                control_check()
                with payload.stream_path.open("xb") as stream:
                    for index in range(payload.output_tokens):
                        stream.write(_sse([200 + index], index + 1))
                    stream.write(_sse([], payload.output_tokens, stop=True))
                links = (row.removeprefix("link:") for row in executed.operator_plan["resource_ids"]
                         if row.startswith("link:"))
                return RawExecutionObservation(
                    started_us=executed.planned_start_us, finished_us=executed.planned_finish_us,
                    output_sha256="f" * 64, payload={"tokens": [200]}, energy=energy(links))

        def provider(_ticket, _at_us):
            """The rig's view: a reaped server leaves the residency map, so its route needs a load."""
            if split in rig._live_executors:
                return hot
            executors = dict(hot.executors)
            executors[split] = replace(executors[split], ready=False, free_slots=0)
            return replace(hot, executors=executors,
                           residency=tuple(row for row in hot.residency if row.executor_id != split))

        backend = Backend()
        payload = LlamaCppCompletionPayload(
            request_id="g1b", expected_model_alias="two-phone", input_tokens=8, output_tokens=30,
            prompt_tokens=tuple(range(8)), seed=0, stream_path=self.root / "request-003.raw",
            on_first_token=lambda _ns: None)
        result = CanonicalPhysicalAdapter(
            scheduler, backend, epoch_ns=epoch_ns, snapshot_provider=provider,
            lease_guard_us=50_000, lease_quantum_us=100_000,
        ).execute(ticket, payload)

        # the attempt ran the real controller: a policy control, k tokens, then the boundary's stats call
        self.assertEqual(calls[1], ("control", False))
        self.assertEqual(calls[-1], ("stats", "g1b"))
        streamed = sum(row[0] == "token" for row in calls)
        # helper_lost and server_exited together: waited for the exit, reaped once
        self.assertEqual(server.waits, [0.5])
        self.assertEqual([(row["kind"], row["executor_id"], row["returncode"]) for row in rig.server_exit_events],
                         [("SERVER_EXITED", split, 1)])
        self.assertNotIn(split, rig._live_executors)
        server.stop.assert_called_once_with()
        # FALLBACK from the prompt on the desktop, through a load (the dead server is never reused)
        self.assertEqual(result.ticket.previous_ticket_id, ticket.ticket_id)
        self.assertEqual(result.ticket.execution_plan.execution_contract.execution_mode, "desktop")
        self.assertNotIn(h.PIXEL, result.ticket.execution_plan.device_ids)
        self.assertEqual(backend.transitions, [row.transition_id for row in result.ticket.execution_plan.transitions])
        self.assertEqual(len(backend.transitions), 1)
        self.assertTrue(backend.transitions[0].startswith("load:"))
        self.assertEqual([row["event_kind"] for row in scheduler.runtime_decision_log()["records"]
                          if row["request_ids"] == ["g1b"]],
                         ["DECISION", "ACQUIRED", "FALLBACK", "ACQUIRED", "COMPLETED"])
        (event,) = result.recovery_events
        self.assertEqual(
            {key: event[key] for key in ("kind", "failure_kind", "device_id", "executor_id", "returncode",
                                         "quarantined_device_ids", "tokens_discarded", "attempt_stream")},
            {"kind": "REQUEST_RECOVERED", "failure_kind": "helper_lost", "device_id": h.PIXEL,
             "executor_id": split, "returncode": 1, "quarantined_device_ids": [h.PIXEL],
             "tokens_discarded": streamed, "attempt_stream": "request-003.raw.attempt1"})
        self.assertEqual(scheduler.quarantined_devices(), {h.PIXEL: "HELPER_LOST"})
        self.assertEqual([(row["kind"], row["device_id"]) for row in scheduler.device_membership_events()],
                         [("DEVICE_QUARANTINED", h.PIXEL)])
        self.assertIn(b"\"stop\": true", payload.stream_path.read_bytes())
        # the membership probe reconciles the loss: the worker is released, the join path takes over
        rig._bind_scheduler(scheduler)
        self.assertEqual(rig._probe_helper_membership(h.PIXEL)["state"], "QUARANTINED")
        lifecycle.wait_released(timeout_s=5)
        self.assertEqual(lifecycle.absent, {h.PIXEL})


class StaticRigUnchangedTests(unittest.TestCase):
    """Without elastic phones the rig registers no membership probe, emits no membership row and starts
    co-helpers exactly as before."""

    def test_static_rig_keeps_its_trace_lifecycle(self):
        calls = []

        class _Lifecycle:
            sessions = {h.PIXEL: object()}

            def start_trace(self, directory):
                calls.append(("start", directory.name))
                return ({"phase": "launch"},)

            def end_trace(self, served):
                calls.append(("end", dict(served)))
                return ({"phase": "stop"},)

        with tempfile.TemporaryDirectory() as directory:
            rig = HeterogeneousPhysicalRig.__new__(HeterogeneousPhysicalRig)
            rig.configuration = SimpleNamespace(output_directory=Path(directory))
            rig._lock, rig._transition_active, rig._execution_backend = threading.RLock(), False, None
            rig._co_helper_lifecycles = {"sha256:" + "e" * 64: _Lifecycle()}
            rig._co_helper_activity = {h.PIXEL: PhoneActivityIntervalTracker()}
            rig._co_helper_receipts = []
            rig.begin_trace(time.monotonic_ns())
            rig._stop_co_helpers()
            self.assertEqual(calls, [("start", "co-helper-eeeeeeeeeeee"), ("end", {})])
            self.assertEqual(rig._helper_membership_observations(), {})
            self.assertFalse((Path(directory) / "HELPER_MEMBERSHIP.json").exists())


class ConstructedRigFlagAbsentTests(unittest.TestCase):
    """T5 through ``HeterogeneousPhysicalRig.__init__`` (process launchers, samplers and the direct phone
    session stubbed): without elastic phones no membership probe is registered, no membership row or
    classifier exists and the RESULT gains no key; with the flag the probes are added and nothing else."""

    BASELINE_PROBES = ["endpoint:physical:desk-cpu", "endpoint:physical:desk-gpu",
                       "endpoint:physical:op15-phone", "helper-runtime:pixel-phone", "phone-runtime"]

    def setUp(self):
        scope = h.TemporaryModel()
        self.model = scope.__enter__()
        self.addCleanup(scope.__exit__)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "bridge").write_bytes(b"bridge")
        self.catalog = h.runtime_catalog(self.model, declaration=h.co_helpers())
        for name in ("LlamaServerProcessLauncher", "LlamaServerProcessConfiguration", "HostEnergySampler",
                     "DirectPhoneFfnSession"):
            patcher = mock.patch.object(rig_module, name)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _rig(self, elastic):
        """A configuration whose file/executable validation is bypassed (``__post_init__`` is not under test)."""
        self.worker = _ElasticWorker()
        configuration = object.__new__(rig_module.HeterogeneousRigConfiguration)
        for key, value in dict(
            catalog=self.catalog, manifests={self.model.model_id: self.model}, server_path=self.root / "server",
            resident_server_path=self.root / "server", model_paths_by_artifact={},
            cuda_library_directory=self.root, resident_library_directory=self.root,
            bridge_path=self.root / "bridge", close_helper_path=self.root / "close", direct_phone_session=None,
            host_metrics=None, phone_diagnostic_endpoint="http://127.0.0.1:1", phone_usb_serial=h.OP15_SERIAL,
            phone_device_id=h.OP15, phone_memory_resource_id="op15-ram", gpu_device_id=h.GPU,
            gpu_memory_resource_id="cuda0", host_memory_resource_id="host", adb_port=5037,
            minimum_usb_speed_mbps=5000, output_directory=self.root, large_phase_id_by_model={},
            transition_phase_id=2, preloaded_model_by_executor={}, active_device_cost_features={},
            resident_model_id=self.model.model_id, resident_executor_id=self.catalog.executors[0].executor_id,
            android_phone_server=None, energy_attribution_kind="diagnostic", host_memory_budget_bytes=None,
            dormant_share_safety_bytes=1, dormant_share_hold_timeout_s=1.0, dormant_share_workspace_bytes=1,
            co_helper_lifecycles={self.model.artifact_sha256: CoHelperLifecycle(
                h.co_helpers(), {h.PIXEL: self.worker}, _Stop(), identity_checks={h.PIXEL: _Identity()})},
            elastic_phones=elastic,
        ).items():
            object.__setattr__(configuration, key, value)
        return HeterogeneousPhysicalRig(configuration, epoch_ns=1)

    def test_flag_absent_rig_registers_no_membership_and_adds_no_result_key(self):
        from research_dev.scheduler.campaigns.burstgpt import runner

        rig = self._rig(None)
        self.assertEqual(sorted(rig._runtime_monitor.probe_names), self.BASELINE_PROBES)
        self.assertIsNone(rig._elastic_phones)
        self.assertFalse(rig._elastic_drop_recovery())
        with mock.patch.object(HeterogeneousPhysicalRig, "_helper_phone_power", return_value={}):
            self.assertIsNone(rig.backend()._failure_classifier)
        rig.begin_trace(time.monotonic_ns())
        self.assertFalse(self.worker.release_failed_start)
        launch = json.loads((self.root / "CO_HELPER_LIFECYCLE.json").read_text())[-1]
        self.assertEqual(set(launch), {"device_id", "phase", "serial", "worker_pids"})
        # a scheduler quarantine is invisible to a static rig: no membership telemetry row, no queue
        scheduler = _Scheduler()
        scheduler.quarantined[h.PIXEL] = "HELPER_LOST"
        rig._bind_scheduler(scheduler)
        self.assertIs(rig._scheduler, scheduler)
        self.assertEqual((rig._helper_membership_observations(), scheduler.calls), ({}, []))
        self.assertEqual((rig.helper_membership_events, rig.server_exit_events), ((), ()))
        self.assertFalse((self.root / "HELPER_MEMBERSHIP.json").exists())
        real = UnifiedScheduler.for_runtime_discovery("enforce")
        real.register_runtime_capabilities(self.catalog)
        self.assertEqual(runner._elastic_drop_result(rig, [{"request_id": "r"}], real), {})

    def test_flagged_rig_adds_only_the_membership_probes(self):
        rig = self._rig(ELASTIC)
        self.assertEqual(sorted(rig._runtime_monitor.probe_names), sorted(
            self.BASELINE_PROBES + ["helper-membership:" + h.PIXEL, "phone-membership:" + h.OP15]))
        with mock.patch.object(HeterogeneousPhysicalRig, "_helper_phone_power", return_value={}):
            self.assertIsNotNone(rig.backend()._failure_classifier)
        rig.begin_trace(time.monotonic_ns())
        self.assertTrue(self.worker.release_failed_start)


# ---- fault injection tool --------------------------------------------------------------------------

class InjectHelperLossToolTests(unittest.TestCase):
    """The fault-injection tool on the launch.py layout: <output>/RESOLVED_CONFIGURATION.json and the
    runner directory <output>/run (``--run-dir``)."""

    ORIGIN_NS = 1_000 * 10**9

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.run_dir = self.output / "run"
        (self.run_dir / "streams").mkdir(parents=True)
        self._resolved({"campaign_id": "eval", "elastic_phones": dict(ELASTIC)})
        self._launches(self._launch(11, self.ORIGIN_NS), self._launch(22, self.ORIGIN_NS + 60 * 10**9))
        self.commands = []

    def _resolved(self, campaign):
        (self.output / tool.RESOLVED_CONFIGURATION_FILE).write_text(json.dumps({"campaign": campaign}))

    @staticmethod
    def _launch(pid, monotonic_ns=None, *, port="5038", root=True):
        """A launch receipt as AdbTcpPhoneWorkerSession.start writes it (the real worker shell command)."""
        configuration = AdbTcpWorkerConfiguration(
            device_id="pixel10pro-phone", serial="SERIAL1", adb_port=int(port or 5037), adb_path=Path("/usr/bin/adb"),
            worker_path=WORKER, library_directories=("/data/local/tmp/w",), shard_path=SHARD,
            artifact_sha256="sha256:" + "a" * 64, layer_mask=0b1100, n_embd=64, columns=128, column_quantum=64,
            max_tokens=4, swiglu=True, backend="CPU", phone_port=26990, worker_environment={"GGML_NTHREADS": "4"},
            expected_sha256_by_path={WORKER: "sha256:" + "a" * 64, SHARD: "sha256:" + "a" * 64},
            as_root=root, phone_lock_path="/data/local/tmp/w/worker.lock" if root else None)
        shell = configuration.worker_shell_command()
        command = ["/usr/bin/adb", *(("-P", port) if port else ()), "-s", "SERIAL1", "shell", "-T", shell]
        return {"phase": "launch", "device_id": "pixel10pro-phone", "serial": "SERIAL1", "worker_pids": [pid],
                "boot_id": "b", "command": command,
                **({} if monotonic_ns is None else {"monotonic_ns": monotonic_ns})}

    def _launches(self, *rows):
        (self.run_dir / tool.LIFECYCLE_FILE).write_text(json.dumps(list(rows)))

    def _run(self, argv, **_options):
        self.commands.append(argv)
        out = self.exe + "\n" if argv[-1].startswith("su -c 'readlink") else ""
        return subprocess.CompletedProcess(argv, 0, out, "")

    exe = WORKER

    def _args(self, *extra):
        return tool.parse_args(["--device", "pixel10pro-phone", "--signal", "TERM",
                                "--run-dir", str(self.run_dir), "--poll-s", "0.01", *extra])

    def test_request_condition_signals_the_latest_worker_with_sigterm(self):
        (self.run_dir / "streams" / "request-012.raw").write_bytes(b"data: {}\n")
        record = tool.inject(self._args("--when", "request=12", "--authorized"), run=self._run)
        self.assertEqual((record["status"], record["signalled_pids"]), ("INJECTED", [22]))
        self.assertEqual([argv[-1] for argv in self.commands],
                         ["su -c 'readlink /proc/22/exe || true'", "su -c 'kill -TERM 22'"])
        # the adb server port of the launch receipt, not the --adb-port default
        self.assertTrue(all(argv[1:3] == ["-P", "5038"] for argv in self.commands))
        saved = json.loads((self.run_dir / tool.RESULT_FILE).read_text())
        self.assertEqual(saved["condition"]["request_index"], 12)
        self.assertEqual((saved["launch_worker_path"], saved["adb_port_source"]), (WORKER, "launch_receipt"))
        self.assertEqual(saved["elastic_phones"], dict(ELASTIC))
        self.assertIn("signal_sent_wall_s", saved["timestamps"])
        with self.assertRaisesRegex(tool.InjectionError, "already exists"):
            tool.inject(self._args("--when", "request=12", "--authorized"), run=self._run)

    def test_only_the_exact_launched_executable_is_signalled(self):
        (self.run_dir / "streams" / "request-001.raw").write_bytes(b"data: {}\n")
        self.exe = "/data/local/tmp/other/llama-ffn-split-worker"
        record = tool.inject(self._args("--when", "request=1", "--authorized"), run=self._run)
        self.assertEqual((record["status"], record["signalled_pids"]), ("NOT_INJECTED", []))
        self.assertFalse(any("kill" in argv[-1] for argv in self.commands))

    def test_adb_port_falls_back_to_the_option_only_without_a_receipt_port(self):
        (self.run_dir / "streams" / "request-001.raw").write_bytes(b"data: {}\n")
        self._launches(self._launch(33, self.ORIGIN_NS, port=None, root=False))
        record = tool.inject(self._args("--when", "request=1", "--authorized"), run=self._run)
        self.assertEqual((record["status"], record["adb_port"]), ("NOT_INJECTED", None))
        self.assertEqual(self.commands, [])
        (self.run_dir / tool.RESULT_FILE).unlink()
        record = tool.inject(self._args("--when", "request=1", "--authorized", "--adb-port", "5040"), run=self._run)
        self.assertEqual((record["status"], record["adb_port_source"], record["signalled_pids"]),
                         ("INJECTED", "--adb-port", [33]))
        self.assertTrue(all(argv[1:3] == ["-P", "5040"] for argv in self.commands))

    def test_time_condition_counts_from_the_first_launch_receipt(self):
        clock = [0.0]

        def sleep(seconds):
            clock[0] += seconds

        # the lifecycle file was rewritten long after the trace started: its mtime is no anchor
        os.utime(self.run_dir / tool.LIFECYCLE_FILE, (1.0, 1.0))
        monotonic = lambda: self.ORIGIN_NS / 1e9 + clock[0]  # noqa: E731
        record = tool.inject(self._args("--when", "t=30", "--authorized", "--timeout-s", "5"), run=self._run,
                             sleep=sleep, monotonic=monotonic, wall_clock=lambda: 10**9 + clock[0])
        self.assertEqual((record["status"], self.commands), ("NOT_INJECTED", []))
        (self.run_dir / tool.RESULT_FILE).unlink()
        clock[0] = 0.0
        record = tool.inject(self._args("--when", "t=3", "--authorized"), run=self._run, sleep=sleep,
                             monotonic=monotonic, wall_clock=lambda: 10**9 + clock[0])
        self.assertEqual(record["status"], "INJECTED")
        self.assertEqual(record["condition"]["trace_origin_monotonic_ns"], self.ORIGIN_NS)
        self.assertGreaterEqual(record["condition"]["elapsed_s"], 3)
        self.assertLess(clock[0], 3.1)

    def test_a_static_phone_run_is_refused(self):
        (self.run_dir / "streams" / "request-001.raw").write_bytes(b"data: {}\n")
        for campaign in ({"campaign_id": "eval"}, {"campaign_id": "eval", "elastic_phones": None}):
            with self.subTest(campaign=campaign):
                self._resolved(campaign)
                with self.assertRaisesRegex(tool.InjectionError, "does not declare elastic_phones"):
                    tool.inject(self._args("--when", "request=1", "--authorized"), run=self._run)
        (self.output / tool.RESOLVED_CONFIGURATION_FILE).unlink()
        with self.assertRaisesRegex(tool.InjectionError, "resolved configuration is unreadable"):
            tool.inject(self._args("--when", "request=1", "--authorized"), run=self._run)
        with mock.patch("sys.stderr"):
            self.assertEqual(tool.main(["--device", "pixel10pro-phone", "--signal", "TERM", "--when", "request=1",
                                        "--run-dir", str(self.run_dir), "--authorized"]), 2)
        self.assertEqual(self.commands, [])
        self.assertFalse((self.run_dir / tool.RESULT_FILE).exists())
        # an explicit resolved configuration elsewhere
        elsewhere = self.output / "elsewhere.json"
        elsewhere.write_text(json.dumps({"campaign": {"elastic_phones": {"join": True}}}))
        record = tool.inject(self._args("--when", "request=1", "--authorized",
                                        "--resolved-configuration", str(elsewhere)), run=self._run)
        self.assertEqual(record["status"], "INJECTED")

    def test_refusals(self):
        with self.assertRaisesRegex(tool.InjectionError, "authorization"):
            tool.inject(self._args("--when", "request=1"), run=self._run)
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            tool.parse_args(["--device", "d", "--signal", "KILL", "--when", "t=1", "--run-dir", str(self.run_dir)])
        for value in ("request=-1", "t=x", "later"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                tool.parse_condition(value)
        self.assertEqual(self.commands, [])


if __name__ == "__main__":
    unittest.main()
