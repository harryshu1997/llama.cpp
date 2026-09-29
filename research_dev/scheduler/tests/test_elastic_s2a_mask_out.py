#!/usr/bin/env python3
"""Elastic phones S2a: a live llama-server that lost one FFN helper is kept and the helper masked out.

``elastic_phones.helper_loss_recovery: "mask_out"`` (reports/20260925-elastic-phones/SPEC.md, S2a).
Fix 3 (``retire``, the default) stops the desktop parent after a helper loss and reloads it (G1g:
60 s penalty). With the S2a server change (S2A_SERVER.diff: a failed helper client whose runtime
policy owns none of its layers is idle, a re-owned one reconnects; announced by
``S41SERVERFFNCAPS``) the rig keeps the live server, the errored requests recover on the same
route of the same process (no transition), policies never give the lost helper a layer there, and
a readmitted helper re-attaches at its next policy (or, on a server that cannot reconnect, when the
server is next relaunched).

llama-server errors EVERY processing slot when a decode aborts (server-context.cpp ``decode``), so
the co-tenants that keep streaming untouched are the ones that were not processing at the loss
(started later, queued); they run on the same live server without any recovery.

Recorded tests, no hardware: fixtures of test_elastic_g1d_recovery (synthetic two-phone catalog,
fake live desktop server, arrival coordinator) and test_per_device_policies (coherent server).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

from research_dev.scheduler import (
    SchedulerConfigurationError,
    UnifiedScheduler,
    elastic_phones_configuration,
)
from research_dev.scheduler._internal.adaptive_decode_contracts import AdaptiveDecodeControl
from research_dev.scheduler._internal.runtime_execution import (
    RuntimeExecutionCoordinatorError,
    RuntimeExecutionFailure,
)
from research_dev.scheduler.adapters import (
    CanonicalArrivalCoordinator,
    CanonicalRuntimeSubmission,
    PhysicalAdapterError,
)
from research_dev.scheduler.adapters.contracts import (
    CompletionStreamError,
    PhysicalBackendFailure,
    PhysicalFailureClassification,
    RawExecutionObservation,
    RawTransitionObservation,
    elastic_helper_loss_recovery,
)
from research_dev.scheduler.adapters.http_backend import (
    CanonicalHttpExecutionBackend,
    LlamaCppCompletionPayload,
    LlamaCppHttpClient,
    _AdaptivePayloadController,
)
from research_dev.scheduler.campaigns.burstgpt.runner import _elastic_drop_result

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
import two_phone_harness as h  # noqa: E402
import test_elastic_g1d_recovery as g1d  # noqa: E402
from test_per_device_policies import (  # noqa: E402
    C, CO_HELPER, CO_HELPER_MASK, HOST, P, PC, PRIMARY, _DeviceSetCase, _Server, devices,
)

MASK_OUT = elastic_phones_configuration(
    {"drop_recovery": True, "join": True, "helper_loss_recovery": "mask_out"})
CAPABILITIES = "S41SERVERFFNCAPS helper_mask_out=1 helper_reconnect_tcp=1"
DESKTOP = g1d.DESKTOP
OP15_LAYERS = 0b001111
# the desktop parent's launch environment: its dormant FFN runtime owns both phones' layers
HELPERS = {
    **g1d.TWO_HELPERS,
    "S41_SERVER_FFN_LAYER_MASK": str(OP15_LAYERS | h.PIXEL_MASK),
    "S41_SERVER_FFN_HELPER0_LAYER_MASK": str(OP15_LAYERS),
    "S41_SERVER_FFN_HELPER1_LAYER_MASK": str(h.PIXEL_MASK),
}
STREAMED = 3


class MaskedServer(g1d.LiveCoTenantServer):
    """The live desktop parent with its launch environment and startup lines; ``lose_helper(label)``
    prints what the S2a server prints when that helper's session fails (a labeled
    ``S41SERVERFFNERROR`` at once, then the aborted decode); unlabeled like the older build."""

    def __init__(self, generation: int, capabilities: str | None) -> None:
        super().__init__(generation)
        self.environment = dict(HELPERS)
        if capabilities is not None:
            self.lines.insert(0, capabilities)
        self.stderr_lines = self.lines
        self.lost_label = "pixel"

    def lose_helper(self, label: str | None = None) -> None:
        if label is not None:
            self.lost_label = label
            with self._condition:
                self.lines.append(f"S41SERVERFFNERROR helper={label} detail=LIBUSB_ERROR_NO_DEVICE")
        super().lose_helper()

    def stop(self) -> None:
        with self._condition:
            self.stop_calls += 1
            if self.returncode is None:
                self.lines.append(f"S41SERVERFFNERROR helper={self.lost_label} detail=session failed")
                self.returncode = 0
            self._condition.notify_all()
        if self.on_stopped is not None:
            self.on_stopped()


def masked_server(generation: int = 1, capabilities: str | None = CAPABILITIES) -> MaskedServer:
    """The live desktop parent; ``capabilities`` is its S2a startup line (None: an older build)."""
    return MaskedServer(generation, capabilities)


def desktop_command(ticket_id: str, mode: str = "desktop") -> SimpleNamespace:
    return SimpleNamespace(ticket_id=ticket_id, executor_id=DESKTOP, request_id=ticket_id.split(":")[0],
                           execution_contract=SimpleNamespace(execution_mode=mode))


def cut() -> CompletionStreamError:
    return CompletionStreamError("completion stream chunk is invalid", server_error_message="Compute aborted.")


def control(policy, request_id: str = "r") -> AdaptiveDecodeControl:
    return AdaptiveDecodeControl(request_id=request_id, slot_id=0, plan_generation=1, policy=policy)


# ---- configuration --------------------------------------------------------------------------------

class HelperLossRecoveryConfigurationTests(unittest.TestCase):
    def test_the_sub_option_is_optional_and_absent_means_retire(self):
        plain = elastic_phones_configuration({"drop_recovery": True, "join": True})
        self.assertNotIn("helper_loss_recovery", plain)
        self.assertEqual(dict(MASK_OUT), {**dict(plain), "helper_loss_recovery": "mask_out"})
        self.assertEqual(elastic_helper_loss_recovery(None), "retire")
        self.assertEqual(elastic_helper_loss_recovery(plain), "retire")
        self.assertEqual(elastic_helper_loss_recovery(MASK_OUT), "mask_out")
        self.assertEqual(elastic_helper_loss_recovery(elastic_phones_configuration(
            {"drop_recovery": True, "helper_loss_recovery": "retire"})), "retire")
        for value in ({"drop_recovery": True, "helper_loss_recovery": "reload"},
                      {"drop_recovery": True, "helper_loss_recovery": True},
                      {"join": True, "helper_loss_recovery": "mask_out"}):
            with self.subTest(value=value), self.assertRaises(SchedulerConfigurationError):
                elastic_phones_configuration(value)

    def test_campaign_arguments_carry_the_sub_option(self):
        from research_dev.scheduler.campaigns.burstgpt.arguments import elastic_phones_json
        from research_dev.scheduler.configuration.campaign import CampaignManifest

        value = elastic_phones_json('{"drop_recovery": true, "join": true, "helper_loss_recovery": "mask_out"}')
        self.assertEqual(value["helper_loss_recovery"], "mask_out")
        # prepare_trace_inputs_v2 --elastic-phones-json writes campaign.elastic_phones verbatim;
        # the campaign manifest validates it and serializes the sub-option only when given
        with tempfile.TemporaryDirectory() as directory:
            for elastic, expected in ((value, "mask_out"), ({"drop_recovery": True, "join": True}, None)):
                manifest = CampaignManifest.from_json({
                    "campaign_id": "elastic", "energy_attribution_kind": "diagnostic",
                    "evidence_manifest_path": "evidence.json", "maximum_latency_ppm": 1_250_000,
                    "models_manifest_path": "models.json", "rig_manifest_path": "rig.json",
                    "schema": "research-scheduler-campaign-v1", "selection_mode": "adaptive-decode",
                    "trace": {"large_requests_path": "l.jsonl", "overlay_requests_path": "o.jsonl",
                              "trace_manifest_path": "t.json"},
                    "elastic_phones": elastic,
                }, Path(directory))
                self.assertEqual(manifest.to_json()["elastic_phones"].get("helper_loss_recovery"), expected)


# ---- failure facts --------------------------------------------------------------------------------

class MaskedFailureFactsTests(unittest.TestCase):
    def test_a_masked_executor_belongs_to_a_helper_lost_failure_of_a_live_server(self):
        facts = RuntimeExecutionFailure("helper_lost", True, True, failed_device_ids=(h.PIXEL,),
                                        masked_executor_id=DESKTOP)
        self.assertTrue(facts.fallback_allowed)
        for arguments in (dict(phase="server_exited", execution_started=False, masked_executor_id=DESKTOP),
                          dict(phase="helper_lost", execution_started=True, failed_device_ids=(h.PIXEL,),
                               masked_executor_id=DESKTOP, exited_executor_id=DESKTOP)):
            with self.subTest(arguments=arguments), self.assertRaises(RuntimeExecutionCoordinatorError):
                RuntimeExecutionFailure(retry_safe=True, **arguments)
        classified = PhysicalFailureClassification("helper_lost", failed_device_ids=(h.PIXEL,),
                                                   masked_executor_id=DESKTOP)
        self.assertEqual(classified.to_json()["masked_executor_id"], DESKTOP)
        self.assertNotIn("masked_executor_id",
                         PhysicalFailureClassification("helper_lost", failed_device_ids=(h.PIXEL,)).to_json())
        with self.assertRaises(PhysicalAdapterError):
            PhysicalFailureClassification("helper_lost", failed_device_ids=(h.PIXEL,), executor_id=DESKTOP,
                                          returncode=0, masked_executor_id=DESKTOP)
        with self.assertRaises(PhysicalAdapterError):
            PhysicalBackendFailure("x", phase="server_exited", retry_safe=True, execution_started=False,
                                   started_us=0, finished_us=1, executor_id=DESKTOP,
                                   masked_executor_id=DESKTOP)


# ---- the rig: classify, mask, guard, re-attach ----------------------------------------------------

class RigMaskOutTests(unittest.TestCase):
    def setUp(self) -> None:
        scope = h.TemporaryModel()
        self.model = scope.__enter__()
        self.addCleanup(scope.__exit__)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.rig = g1d.partial_rig(Path(directory.name), self.model, time.monotonic_ns())
        self.rig.configuration.elastic_phones = MASK_OUT
        self.rig._failure_evidence_timeout_s = 5.0
        self.server = masked_server()
        g1d.publish(self.rig, self.model, self.server)
        from test_per_device_policies import device_policy

        self.op15_only = device_policy(((h.OP15, OP15_LAYERS),))
        self.pixel_only = device_policy(((h.PIXEL, h.PIXEL_MASK),))
        self.both = device_policy(((h.OP15, OP15_LAYERS), (h.PIXEL, h.PIXEL_MASK)))
        self.baseline = g1d_baseline()

    def mark(self, ticket_id: str) -> None:
        self.rig._execution_markers[ticket_id] = (
            self.server, None, SimpleNamespace(stderr_index=len(self.server.lines)), (8, 30))

    def classify(self, ticket_id: str, started_ns: int | None, mode: str = "desktop"):
        return self.rig._classify_execution_failure(desktop_command(ticket_id, mode), cut(), started_ns=started_ns)

    def lose(self, labeled: bool = False) -> None:
        if labeled:
            # the S2a server names the helper the moment its session fails
            self.server.lines.append("S41SERVERFFNERROR helper=pixel detail=FFN split EXECUTE header exchange failed")
        self.server.lose_helper()

    def test_co_tenants_of_one_loss_mask_the_helper_once_and_keep_the_server(self):
        started = time.monotonic_ns()
        for labeled in (False, True):
            with self.subTest(labeled=labeled):
                self.setUp()
                tickets = [f"t{index}:attempt:0" for index in range(g1d.CO_TENANTS)]
                for ticket in tickets:
                    self.mark(ticket)
                self.lose(labeled)
                results = [None] * len(tickets)
                threads = [threading.Thread(target=lambda index=index: results.__setitem__(
                    index, self.classify(tickets[index], started))) for index in range(len(tickets))]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(10)
                self.assertEqual(
                    {(row.phase, row.failed_device_ids, row.executor_id, row.returncode, row.masked_executor_id)
                     for row in results},
                    {("helper_lost", (h.PIXEL,), None, None, DESKTOP)})
                self.assertEqual(self.server.stop_calls, 0)
                self.assertIs(self.rig._live_executors[DESKTOP].server, self.server)
                self.assertEqual(self.rig.server_exit_events, ())
                (masked,) = self.rig.helper_mask_events
                self.assertEqual({key: masked[key] for key in ("kind", "server", "device", "layer_mask", "generation")},
                                 {"kind": "HELPER_MASKED_OUT", "server": DESKTOP, "device": h.PIXEL,
                                  "layer_mask": h.PIXEL_MASK, "generation": 1})
                self.assertEqual(self.rig.masked_helper_devices(DESKTOP), (h.PIXEL,))

    def test_mask_out_is_refused_and_the_server_retired_when_it_cannot_stay_clean(self):
        cases = {
            "SERVER_LACKS_HELPER_MASK_OUT": dict(capabilities=None),
            "NOT_A_DESKTOP_PARENT": dict(mode="adaptive-split"),
            "MASKED_HELPER_LOST_AGAIN": dict(again=True),
        }
        for reason, case in cases.items():
            with self.subTest(reason=reason):
                self.setUp()
                if "capabilities" in case:
                    self.server = masked_server(capabilities=None)
                    g1d.publish(self.rig, self.model, self.server)
                if case.get("again"):
                    self.mark("first:attempt:0")
                    self.lose()
                    self.assertEqual(self.classify("first:attempt:0", time.monotonic_ns()).masked_executor_id,
                                     DESKTOP)
                    # an attempt that started after the mask sees the masked helper fail again
                    later = time.monotonic_ns()
                    self.mark("later:attempt:0")
                    self.lose()
                    classification = self.classify("later:attempt:0", later)
                else:
                    self.mark("t0:attempt:0")
                    self.lose()
                    classification = self.classify("t0:attempt:0", time.monotonic_ns(), case.get("mode", "desktop"))
                self.assertEqual((classification.masked_executor_id, classification.executor_id,
                                  classification.returncode), (None, DESKTOP, 0))
                self.assertEqual(self.server.stop_calls, 1)
                (exited,) = self.rig.server_exit_events
                self.assertEqual(exited["cause"], "HELPER_LOST")
                refused = [row for row in self.rig.helper_mask_events if row["kind"] == "HELPER_MASK_OUT_REFUSED"]
                self.assertEqual([(row["device"], row["reason"]) for row in refused], [(h.PIXEL, reason)])
                if case.get("again"):
                    # the retire ended the mask with the server
                    self.assertEqual([row["kind"] for row in self.rig.helper_mask_events][-1], "HELPER_MASK_ENDED")
                    self.assertEqual(self.rig.masked_helper_devices(DESKTOP), ())

    def test_retire_mode_and_an_absent_flag_keep_todays_retire(self):
        for elastic in (g1d.ELASTIC, elastic_phones_configuration(
                {"drop_recovery": True, "helper_loss_recovery": "retire"})):
            with self.subTest(elastic=dict(elastic)):
                self.setUp()
                self.rig.configuration.elastic_phones = elastic
                self.mark("t0:attempt:0")
                self.lose()
                classification = self.classify("t0:attempt:0", time.monotonic_ns())
                self.assertEqual((classification.masked_executor_id, classification.returncode), (None, 0))
                self.assertEqual(self.server.stop_calls, 1)
                self.assertEqual(self.rig.helper_mask_events, ())

    def test_the_guard_drops_every_policy_that_gives_the_masked_helper_a_layer(self):
        self.mark("t0:attempt:0")
        self.lose()
        self.classify("t0:attempt:0", time.monotonic_ns())
        command = desktop_command("r1:attempt:1")
        for policy in (self.op15_only, self.baseline):
            self.rig._helper_mask_guard(command, control(policy))
        # by owner, and by layer even when a policy carries no owners (one-phone form)
        single_owner = replace(self.pixel_only, device_layer_masks=())
        for policy in (self.pixel_only, self.both, single_owner):
            with self.subTest(policy=policy.route_id), self.assertRaisesRegex(
                    PhysicalAdapterError, "quarantined helper " + h.PIXEL):
                self.rig._helper_mask_guard(command, control(policy))
        dropped = [row for row in self.rig.helper_mask_events if row["kind"] == "HELPER_POLICY_DROPPED"]
        self.assertEqual([row["reason"] for row in dropped], ["MASKED_OUT"] * 3)
        # another server is not affected by this server's mask ...
        other = SimpleNamespace(executor_id="physical:other", request_id="r")
        self.rig._helper_mask_guard(other, control(self.both))
        # ... but no server may give a device the scheduler holds quarantined a layer
        self.rig._scheduler = SimpleNamespace(quarantined_devices=lambda: {h.PIXEL: "HELPER_LOST"})
        with self.assertRaisesRegex(PhysicalAdapterError, "quarantined helper " + h.PIXEL):
            self.rig._helper_mask_guard(other, control(self.both))
        self.rig._helper_mask_guard(other, control(self.op15_only))
        self.assertEqual(self.rig.helper_mask_events[-1]["reason"], "QUARANTINED")

    def test_a_readmitted_helper_reattaches_live_or_waits_for_the_relaunch(self):
        for reconnect in (True, False):
            with self.subTest(reconnect=reconnect):
                self.setUp()
                if not reconnect:
                    self.server = masked_server(capabilities="S41SERVERFFNCAPS helper_mask_out=1")
                    g1d.publish(self.rig, self.model, self.server)
                self.mark("t0:attempt:0")
                self.lose()
                self.classify("t0:attempt:0", time.monotonic_ns())
                self.rig._reattach_masked_helpers(h.OP15)  # not masked: nothing happens
                self.rig._reattach_masked_helpers(h.PIXEL)
                kinds = [row["kind"] for row in self.rig.helper_mask_events]
                if reconnect:
                    # the Pixel is a TCP helper of a server that reconnects TCP helpers
                    self.assertEqual(kinds, ["HELPER_MASKED_OUT", "HELPER_REATTACHED"])
                    self.assertEqual(self.rig.helper_mask_events[-1]["mode"], "live_reconnect")
                    self.rig._helper_mask_guard(desktop_command("r2:attempt:0"), control(self.both))
                    self.assertEqual(self.rig.masked_helper_devices(DESKTOP), ())
                    continue
                self.assertEqual(kinds, ["HELPER_MASKED_OUT", "HELPER_REATTACH_PENDING_RELAUNCH"])
                self.assertEqual(self.rig.helper_mask_events[-1]["reason"], "SERVER_CANNOT_RECONNECT")
                self.rig._reattach_masked_helpers(h.PIXEL)  # idempotent
                self.assertEqual(len(self.rig.helper_mask_events), 2)
                with self.assertRaises(PhysicalAdapterError):
                    self.rig._helper_mask_guard(desktop_command("r2:attempt:0"), control(self.both))
                # the next planned stop (model switch) ends the mask; the relaunch connects afresh
                self.rig._stop_executor(DESKTOP, terminate_phone_session=False)
                ended = self.rig.helper_mask_events[-1]
                self.assertEqual((ended["kind"], ended["cause"], ended["pending_relaunch"]),
                                 ("HELPER_MASK_ENDED", "SERVER_STOPPED", True))
                relaunched = masked_server(2)
                g1d.publish(self.rig, self.model, relaunched)
                self.rig._helper_mask_guard(desktop_command("r3:attempt:0"), control(self.both))

    def test_a_replaced_server_drops_its_stale_masks(self):
        self.mark("t0:attempt:0")
        self.lose()
        self.classify("t0:attempt:0", time.monotonic_ns())
        g1d.publish(self.rig, self.model, masked_server(2))  # published without a stop callback
        self.rig._helper_mask_guard(desktop_command("r2:attempt:0"), control(self.both))
        self.assertEqual(self.rig.helper_mask_events[-1]["cause"], "SERVER_REPLACED")


def g1d_baseline():
    from test_adaptive_decode import policy

    return policy("desktop-control", 0, (), baseline=True)


class HttpControlGuardTests(unittest.TestCase):
    """The HTTP backend asks the rig's guard before any FFN control reaches the server."""

    def controller(self, guard):
        sent = []

        class Client(LlamaCppHttpClient):
            def apply_ffn_control(self, endpoint, value, *, timeout_s=5):
                sent.append(value.policy)
                return {"ok": True}, 1

        backend = CanonicalHttpExecutionBackend(
            Client(), SimpleNamespace(measure=lambda *_args: None), epoch_ns=0, control_guard=guard)
        controller = _AdaptivePayloadController.__new__(_AdaptivePayloadController)
        controller.backend, controller.cohort = backend, None
        controller.command = SimpleNamespace(request_id="r", ticket_id="r:attempt:1", endpoint="http://127.0.0.1:1",
                                             executor_id=DESKTOP, planned_start_us=0, planned_finish_upper_us=1)
        controller.payload = SimpleNamespace(timeout_s=30)
        return controller, sent

    def test_a_refused_control_never_reaches_the_server(self):
        from test_per_device_policies import device_policy

        pixel = device_policy(((h.PIXEL, h.PIXEL_MASK),))
        op15 = device_policy(((h.OP15, OP15_LAYERS),))

        def guard(command, value):
            if h.PIXEL in devices(value.policy):
                raise PhysicalAdapterError("FFN control gives masked-out helper " + h.PIXEL)

        controller, sent = self.controller(guard)
        with self.assertRaisesRegex(PhysicalAdapterError, "masked-out helper"):
            controller._send_control(control(pixel), ())
        controller._send_control(control(op15), ())
        self.assertEqual(sent, [op15])
        with self.assertRaisesRegex(PhysicalAdapterError, "control guard is invalid"):
            CanonicalHttpExecutionBackend(LlamaCppHttpClient(), SimpleNamespace(measure=lambda *_args: None),
                                          epoch_ns=0, control_guard=object())


# ---- the coherent server policy after the quarantine ----------------------------------------------

class MaskedServerPolicyTests(_DeviceSetCase):
    """The server policy applied to the masked server's sessions after the quarantine excludes the
    lost helper for every batch composition, and each live session's very next control does."""

    def test_every_composition_and_every_next_control_exclude_the_masked_helper(self):
        server = _Server(self, self.energies, self.latencies)
        policies = []
        acknowledge = server._acknowledge

        def recording(request_id, directive):
            if directive.control is not None:
                policies.append((request_id, directive.control.policy))
            return acknowledge(request_id, directive)

        server._acknowledge = recording
        group = self.qualify_both_at_batch_one(server)
        # a co-tenant joins: batch 2 is probed from the batch-1 verdict (primary + co-helper)
        server.start("request-b", 2, active_batch=2)
        server.tick(next_active_batch={"request-a": 2})
        self.assertIn(PC, {devices(row) for _batch, row in group.verdicts})
        del policies[:]
        server.controller.quarantine_device(CO_HELPER, reason="HELPER_LOST", at_us=server.now)
        # every composition: no verdict, proposal or running policy keeps the lost helper
        for _batch, verdict in group.verdicts:
            self.assertNotIn(CO_HELPER, devices(verdict))
        self.assertTrue(group.proposal is None or CO_HELPER not in devices(group.proposal))
        self.assertNotIn(CO_HELPER, devices(group.policy))
        server.tick()
        self.assertEqual({request for request, _policy in policies}, {"request-a", "request-b"})
        for _ in range(4):
            server.tick()
        self.assertTrue(policies)
        for request_id, policy in policies:
            self.assertNotIn(CO_HELPER, devices(policy))
            self.assertEqual(policy.layer_mask & CO_HELPER_MASK, 0, request_id)
        self.assertEqual({server.running(row) for row in ("request-a", "request-b")} - {HOST}, {P})


# ---- end to end: the recovered requests stay on the live server -----------------------------------

class MaskOutScenario:
    """Two requests stream on the live desktop parent when the Pixel dies (both errored by the
    server); a third was acquired with them but starts only after the loss (not errored). The rig
    keeps the server; recoveries run on the same process, without a transition."""

    STREAMING = 2

    def __init__(self, case: unittest.TestCase, *, adaptive: bool = False,
                 capabilities: str | None = CAPABILITIES, elastic=MASK_OUT,
                 lost_label: str | None = None) -> None:
        scope = h.TemporaryModel()
        self.model = scope.__enter__()
        case.addCleanup(scope.__exit__)
        directory = tempfile.TemporaryDirectory()
        case.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.adaptive = adaptive
        self.catalog = g1d.co_tenant_catalog(self.model)
        self.scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        self.scheduler.register_runtime_capabilities(self.catalog)
        self.scheduler.register_model_manifest(self.model)
        self.hot = g1d.hot_snapshot(self.model, self.catalog)
        if adaptive:
            self.hot = g1d.with_dormant_parent(self.model, self.catalog, self.hot)
        self.cold = g1d.reaped_snapshot(self.hot)
        self.epoch_ns = time.monotonic_ns()
        self.rig = g1d.partial_rig(self.root, self.model, self.epoch_ns)
        self.rig.configuration.elastic_phones = elastic
        self.rig._failure_evidence_timeout_s = 30.0
        self.lost_label = lost_label
        self.server = masked_server(capabilities=capabilities)
        g1d.publish(self.rig, self.model, self.server)
        self.lost = threading.Event()
        self.in_flight = threading.Barrier(self.STREAMING, action=self._lose, timeout=20)
        self.http = CanonicalHttpExecutionBackend(
            self._client(), SimpleNamespace(measure=lambda *_args: None), epoch_ns=self.epoch_ns,
            on_execution_start=self._begin, on_execution_finish=self._finish,
            failure_classifier=self.rig._classify_execution_failure, control_guard=self.rig._helper_mask_guard,
        )
        self.attempts, self.ran, self.transitions, self.routes = {}, [], [], {}
        self.transition_parameters = []
        self.controls, self.sessions = [], {}

    def _lose(self) -> None:
        self.server.lose_helper(self.lost_label)
        self.lost.set()

    def now_us(self) -> int:
        return (time.monotonic_ns() - self.epoch_ns) // 1_000

    def provider(self, _ticket, _at_us):
        """The rig's view: the live server is hot; a retired one left the residency map."""
        with self.rig._lock:
            return self.hot if DESKTOP in self.rig._live_executors else self.cold

    def _client(self):
        scenario = self

        class StreamingClient(LlamaCppHttpClient):
            def complete(self, endpoint, payload, control_check, **_options):
                with payload.stream_path.open("xb") as stream:
                    for index in range(STREAMED):
                        stream.write(g1d._sse([100 + index], index + 1))
                payload.on_first_token(time.monotonic_ns())
                if scenario.adaptive:
                    scenario.start(payload.request_id)
                scenario.in_flight.wait()
                raise cut()

        return StreamingClient()

    def _begin(self, command) -> None:
        with self.rig._lock:
            state = self.rig._live_executors.get(command.executor_id)
        self.rig._require_unretired_endpoint(command.executor_id, state)
        with self.rig._lock:
            self.rig._execution_markers[command.ticket_id] = (
                state.server, None, SimpleNamespace(stderr_index=len(state.server.lines)), (8, 30))

    def _finish(self, command) -> None:
        with self.rig._lock:
            self.rig._execution_markers.pop(command.ticket_id)

    def start(self, request_id):
        from research_dev.scheduler import AdaptiveDecodeConfig

        directive = self.scheduler.start_adaptive_decode(
            request_id, slot_id=0, first_token_index=1, at_us=self.now_us(),
            config=AdaptiveDecodeConfig(**g1d.G1fRecoveredAdaptiveAttempt.WINDOW))
        self.sessions.setdefault(request_id, []).append(self.scheduler.runtime_ticket(request_id).ticket_id)
        return directive

    def _decode(self, command, payload) -> None:
        """A request on the live server after the loss: its adaptive session (if any) runs to the end;
        every control it asks the server for passes the rig's guard first, as in the HTTP backend."""
        if self.adaptive and CanonicalHttpExecutionBackend._adaptive_enabled(command):
            directive = self.start(command.request_id)
            while directive is not None:
                if directive.control is not None:
                    self.controls.append((command.request_id, directive.control.policy))
                    self.rig._helper_mask_guard(command, directive.control)
                if directive.target_token_index is None:
                    break
                directive = g1d.G1dScenario.adaptive_window(self, command.request_id, directive)
        with payload.stream_path.open("xb") as stream:
            for index in range(payload.output_tokens):
                stream.write(g1d._sse([200 + index], index + 1))
            stream.write(g1d._sse([], payload.output_tokens, stop=True))
        if self.adaptive and CanonicalHttpExecutionBackend._adaptive_enabled(command):
            self.scheduler.complete_adaptive_decode(command.request_id)

    def backend(self):
        scenario = self

        class Backend:
            def bind_scheduler(self, bound):
                scenario.http.bind_scheduler(bound)

            def apply_transition(self, transition, payload, control_check):
                control_check()
                scenario.transitions.append(transition.transition.transition_id)
                scenario.transition_parameters.append(dict(transition.adapter_parameters))
                with scenario.rig._lock:
                    if DESKTOP not in scenario.rig._live_executors:
                        g1d.publish(scenario.rig, scenario.model, masked_server(2))
                return RawTransitionObservation(started_us=scenario.now_us(), finished_us=scenario.now_us() + 1,
                                                status="COMPLETED", evicted_artifact_sha256s=())

            def execute(self, command, payload, control_check):
                attempt = scenario.attempts[command.request_id] = scenario.attempts.get(command.request_id, 0) + 1
                scenario.routes[(command.request_id, attempt)] = command.route_id
                if attempt == 1 and command.request_id != "g1d-2":
                    return scenario.http.execute(command, payload, control_check)
                if attempt == 1:
                    # acquired with the others, starts after the loss was handled (masked or retired)
                    scenario.lost.wait(10)
                    deadline = time.monotonic() + 10
                    while not (scenario.rig.helper_mask_events or scenario.rig.server_exit_events):
                        if time.monotonic() > deadline:
                            raise AssertionError("the loss was never handled")
                        time.sleep(0.01)
                    with scenario.rig._lock:
                        state = scenario.rig._live_executors.get(command.executor_id)
                    if state is None or state.server is not scenario.server:
                        # a retired endpoint: the HTTP backend's start refuses it (server_exited)
                        return scenario.http.execute(command, payload, control_check)
                control_check()
                with scenario.rig._lock:
                    state = scenario.rig._live_executors[command.executor_id]
                scenario.ran.append((command.request_id, attempt, state.server.generation,
                                     state.server is scenario.server))
                scenario._decode(command, payload)
                links = (row.removeprefix("link:") for row in command.operator_plan["resource_ids"]
                         if row.startswith("link:"))
                return RawExecutionObservation(
                    started_us=command.planned_start_us, finished_us=command.planned_finish_us,
                    output_sha256="f" * 64, payload={"tokens": [200]},
                    energy=g1d.G1dScenario.energy(scenario, links))

        return Backend()

    def payload(self, index: int):
        return LlamaCppCompletionPayload(
            request_id=f"g1d-{index}", expected_model_alias="two-phone", input_tokens=8,
            output_tokens=30, prompt_tokens=tuple(range(8)), seed=0,
            stream_path=self.root / f"request-00{index}.raw", on_first_token=lambda _ns: None)

    def run(self):
        coordinator = CanonicalArrivalCoordinator(
            self.scheduler, self.backend(), epoch_ns=self.epoch_ns,
            snapshot_provider=self.provider, max_workers=3,
            lease_guard_us=50_000, lease_quantum_us=100_000)
        self.payloads = [self.payload(index) for index in range(3)]
        try:
            arrival_us = h.request("g1d-0").arrival_us
            coordinator.wait_for_arrival(arrival_us)
            for index in range(3):
                coordinator.submit(CanonicalRuntimeSubmission(
                    h.request(f"g1d-{index}"), self.model.model_id, self.hot, self.payloads[index],
                ), observed_at_us=arrival_us)
            return coordinator.drain(timeout_s=60)
        finally:
            coordinator.close(wait=False)

    def events(self, request_id: str):
        return [row for row in self.scheduler.runtime_decision_log()["records"]
                if row["request_ids"] == [request_id]]


class SameServerRecoveryTests(unittest.TestCase):
    def assert_same_server_recovery(self, scenario, completed, device=h.PIXEL, layer_mask=h.PIXEL_MASK):
        errored, untouched = ("g1d-0", "g1d-1"), "g1d-2"
        self.assertEqual(completed.request_ids, (*errored, untouched))
        for request_id in (*errored, untouched):
            self.assertEqual(scenario.scheduler.runtime_ticket(request_id).dispatch_state, "COMPLETED")
        # the live server was kept: no stop, no SERVER_EXITED, no load, every attempt on generation 1
        self.assertEqual(scenario.server.stop_calls, 0)
        self.assertEqual(scenario.rig.server_exit_events, ())
        self.assertEqual(scenario.transitions, [])
        self.assertEqual(sorted(scenario.ran),
                         [("g1d-0", 2, 1, True), ("g1d-1", 2, 1, True), (untouched, 1, 1, True)])
        for index, request_id in enumerate(errored):
            result = completed.executions[request_id]
            (event,) = result.recovery_events
            self.assertEqual(
                {key: event[key] for key in ("kind", "failure_kind", "device_id", "executor_id", "returncode",
                                             "masked_executor_id", "recovery", "tokens_discarded",
                                             "quarantined_device_ids", "attempt_stream")},
                {"kind": "REQUEST_RECOVERED", "failure_kind": "helper_lost", "device_id": device,
                 "executor_id": None, "returncode": None, "masked_executor_id": DESKTOP,
                 "recovery": "same_server_mask_out", "tokens_discarded": STREAMED,
                 "quarantined_device_ids": [device], "attempt_stream": f"request-00{index}.raw.attempt1"})
            # no retire, no reload: the penalty is the failed attempt itself (G1g: 60.6 s)
            self.assertLess(event["penalty_us"], 5_000_000)
            ticket = result.ticket
            self.assertEqual((ticket.binding.executor_id, ticket.execution_plan.transitions),
                             (DESKTOP, ()))
            self.assertEqual(scenario.routes[(request_id, 2)], scenario.routes[(request_id, 1)])
            self.assertEqual(ticket.decision.route_id, scenario.routes[(request_id, 1)])
            self.assertNotIn(device, ticket.execution_plan.device_ids)
            kinds = [row["event_kind"] for row in scenario.events(request_id)]
            self.assertEqual(kinds[:3], ["DECISION", "ACQUIRED", "FALLBACK"])
            self.assertEqual(kinds[-1], "COMPLETED")
            fallback = next(row for row in scenario.events(request_id) if row["event_kind"] == "FALLBACK")
            self.assertEqual(fallback["decision_reason"], "PAIRED_DESKTOP_RECOVERY")
            self.assertIn(b"\"stop\": true", scenario.payloads[index].stream_path.read_bytes())
        self.assertEqual(completed.executions[untouched].recovery_events, ())
        self.assertEqual([row["event_kind"] for row in scenario.events(untouched)],
                         ["DECISION", "ACQUIRED", "COMPLETED"])
        self.assertEqual([(row["kind"], row["device_id"]) for row in scenario.scheduler.device_membership_events()],
                         [("DEVICE_QUARANTINED", device)])
        self.assertEqual([(row["kind"], row["server"], row["device"], row["layer_mask"])
                          for row in scenario.rig.helper_mask_events if row["kind"].startswith("HELPER_")],
                         [("HELPER_MASKED_OUT", DESKTOP, device, layer_mask)])
        # RESULT: mask and recovery rows, no SERVER_EXITED
        rows = [{"request_recovered": [dict(row) for row in completed.executions[rid].recovery_events]}
                for rid in completed.request_ids]
        exported = _elastic_drop_result(scenario.rig, rows, scenario.scheduler)
        self.assertEqual(set(exported), {"request_recovered_events", "device_membership_events",
                                         "helper_mask_events"})
        self.assertEqual({row["recovery"] for row in exported["request_recovered_events"]},
                         {"same_server_mask_out"})

    def test_errored_requests_recover_on_the_same_live_server_without_a_reload(self):
        scenario = MaskOutScenario(self)
        started = time.monotonic()
        completed = scenario.run()
        self.assert_same_server_recovery(scenario, completed)
        self.assertLess(time.monotonic() - started, scenario.rig._failure_evidence_timeout_s)

    def test_adaptive_sessions_on_the_dormant_parent_never_give_the_pixel_a_layer(self):
        scenario = MaskOutScenario(self, adaptive=True)
        completed = scenario.run()
        self.assert_same_server_recovery(scenario, completed)
        for request_id in ("g1d-0", "g1d-1"):
            # the failed attempt's session was replaced by the recovered attempt's on the same server
            attempts = scenario.sessions[request_id]
            self.assertEqual(len(attempts), 2)
            grouped = scenario.scheduler.adaptive_decode_grouped_observation(request_id)
            self.assertEqual((grouped.ticket_id, grouped.terminal_status), (attempts[1], "COMPLETED"))
        for _request_id, policy in scenario.controls:
            self.assertNotIn(h.PIXEL, devices(policy))
            self.assertEqual(policy.layer_mask & h.PIXEL_MASK, 0)
        self.assertEqual([row for row in scenario.rig.helper_mask_events if row["kind"] == "HELPER_POLICY_DROPPED"],
                         [])

    def test_without_the_server_capability_the_server_is_retired_as_before(self):
        """An older llama-server build (no S41SERVERFFNCAPS) stays poisoned by the failed client:
        mask_out falls back to fix 3 (retire + one reload), recorded as HELPER_MASK_OUT_REFUSED."""
        scenario = MaskOutScenario(self, capabilities=None)
        completed = scenario.run()
        self.assertEqual(len(completed.request_ids), 3)
        self.assertEqual(scenario.server.stop_calls, 1)
        (exited,) = scenario.rig.server_exit_events
        self.assertEqual(exited["cause"], "HELPER_LOST")
        refused = {row["reason"] for row in scenario.rig.helper_mask_events
                   if row["kind"] == "HELPER_MASK_OUT_REFUSED"}
        self.assertEqual(refused, {"SERVER_LACKS_HELPER_MASK_OUT"})
        self.assertNotIn("HELPER_MASKED_OUT", {row["kind"] for row in scenario.rig.helper_mask_events})
        recovered = [row for rid in completed.request_ids for row in completed.executions[rid].recovery_events]
        self.assertTrue(recovered)
        self.assertTrue(all("recovery" not in row and "masked_executor_id" not in row for row in recovered))
        self.assertEqual(len([row for row in scenario.transitions if row.startswith("load:")]), 1)
        self.assertTrue(all(generation == 2 for rid, attempt, generation, _same in scenario.ran
                            if (rid, attempt) != ("g1d-2", 1)))


class PrimaryLossTests(unittest.TestCase):
    """G2: the OP15 (primary, FunctionFS) is lost on the two-helper desktop parent. The S2a server
    names it at once (``S41SERVERFFNERROR helper=op15``); the recovery never attaches the OP15
    session, and the Pixel alone may still assist."""

    def test_mask_out_keeps_the_parent_and_leaves_the_co_helper_alone(self):
        from test_per_device_policies import device_policy

        scenario = MaskOutScenario(self, lost_label="op15")
        completed = scenario.run()
        SameServerRecoveryTests.assert_same_server_recovery(self, scenario, completed, h.OP15, OP15_LAYERS)
        required = [row for row in scenario.rig.helper_mask_events
                    if row["kind"] == "PRIMARY_SESSION_RELAUNCH_REQUIRED"]
        self.assertEqual([(row["server"], row["device"]) for row in required], [(DESKTOP, h.OP15)])
        self.assertTrue(scenario.rig._direct_phone_relaunch_pending())
        (masked,) = [row for row in scenario.rig.helper_mask_events if row["kind"] == "HELPER_MASKED_OUT"]
        self.assertEqual(masked["transport"], "functionfs-usb")
        command = SimpleNamespace(executor_id=DESKTOP, request_id="next")
        scenario.rig._helper_mask_guard(command, control(device_policy(((h.PIXEL, h.PIXEL_MASK),))))
        for owners in (((h.OP15, OP15_LAYERS),), ((h.OP15, OP15_LAYERS), (h.PIXEL, h.PIXEL_MASK))):
            with self.assertRaises(PhysicalAdapterError):
                scenario.rig._helper_mask_guard(command, control(device_policy(owners)))

    def test_retire_mode_relaunches_the_parent_without_the_primary_session(self):
        scenario = MaskOutScenario(self, elastic=g1d.ELASTIC, lost_label="op15")
        completed = scenario.run()
        self.assertEqual(len(completed.request_ids), 3)
        self.assertEqual(scenario.server.stop_calls, 1)
        (exited,) = scenario.rig.server_exit_events
        self.assertEqual((exited["cause"], exited["failed_device_ids"]), ("HELPER_LOST", [h.OP15]))
        (load,) = scenario.transition_parameters
        # the rig attaches the OP15 session only for a command that names a phone
        # (heterogeneous_rig_ops/transitions.py _prepare_transition_phone): the relaunched parent
        # keeps its helpers deferred until a policy owns their layers
        self.assertNotIn("phone_device_id", load)
        self.assertEqual({(rid, generation) for rid, _attempt, generation, _same in scenario.ran},
                         {("g1d-0", 2), ("g1d-1", 2), ("g1d-2", 2)})
        for request_id in completed.request_ids:
            self.assertNotIn(h.OP15, completed.executions[request_id].ticket.execution_plan.device_ids)
        self.assertEqual(dict(scenario.scheduler.quarantined_devices()), {h.OP15: "HELPER_LOST"})
        self.assertEqual(scenario.rig.helper_mask_events, ())

    def test_a_masked_primary_opens_the_co_helper_alone_on_the_server(self):
        case = MaskedServerPolicyTests("test_every_composition_and_every_next_control_exclude_the_masked_helper")
        case.setUp()
        server = _Server(case, case.energies, case.latencies)
        case.qualify_both_at_batch_one(server)
        del server.controls[:]
        server.controller.quarantine_device(PRIMARY, reason="HELPER_LOST", at_us=server.now)
        for _ in range(4):
            server.tick()
        self.assertEqual(server.running(), C)
        self.assertTrue(server.controls)
        self.assertTrue(all(PRIMARY not in owners for _request, owners in server.controls))


# ---- G2: the primary phone rejoins only on its pinned kernel ---------------------------------------

STOCK_KERNEL = "6.12.23-android16-5-gb3b66ace21e0-ab14672634-4k"
PINNED_KERNEL = "6.12.23-android16-5-o-g227664cbe007-4k"


class PrimaryRejoinKernelTests(unittest.TestCase):
    """The OP15 rebooted mid-run returns on its stock kernel: its readmission waits for the pinned
    kernel release (the FunctionFS transport identity refuses any other at session launch)."""

    def setUp(self):
        import test_elastic_join as join

        case = join._RigMembershipCase("run")
        case.setUp()
        self.addCleanup(case.doCleanups)
        directory = Path(tempfile.mkdtemp(dir=case.root))
        self.release, self.log, self.adb = directory / "release", directory / "adb.log", directory / "adb"
        # a fake adb: prints the phone's `uname -r` and records its arguments and its stdin
        self.adb.write_text("#!/bin/sh\n"
                            f"echo \"$* stdin=$(readlink /proc/$$/fd/0)\" >> {self.log}\n"
                            f"[ -s {self.release} ] || {{ echo 'error: device offline' >&2; exit 1; }}\n"
                            f"cat {self.release}\n")
        self.adb.chmod(0o755)
        rig, _lifecycle, _identity = case._rig(join._ElasticWorker())
        rig.begin_trace(time.monotonic_ns())
        rig.configuration.direct_phone_session = SimpleNamespace(
            required_kernel_release=PINNED_KERNEL, adb_path=self.adb, adb_port=5037, serial=h.OP15_SERIAL)
        self.scheduler = join._Scheduler()
        rig._bind_scheduler(self.scheduler)
        self.scheduler.quarantined[h.OP15] = "HELPER_LOST"
        self.rig = rig
        self.restored = SimpleNamespace(serial=h.OP15_SERIAL, vendor_id="18d1", product_id="4ee7")

    def probe(self):
        from unittest import mock

        if h.OP15 in self.rig._membership:
            self.rig._membership[h.OP15].last_join_ns = 0
        with mock.patch.object(g1d.rig_module, "verify_android_usb_restored", return_value=self.restored):
            return self.rig._probe_primary_membership()["state"]

    def rejections(self):
        return [row for row in self.rig.helper_membership_events if row["kind"] == "JOIN_REJECTED"]

    def test_a_stock_kernel_is_refused_until_the_pinned_kernel_returns(self):
        from research_dev.scheduler._internal.types import canonical_sha256

        self.release.write_text(STOCK_KERNEL + "\n")
        for _ in range(2):
            self.assertEqual(self.probe(), "QUARANTINED")
        (rejected,) = self.rejections()  # one row per distinct reason; every due join retries
        self.assertEqual({key: rejected[key] for key in ("device_id", "reason", "observed", "pinned")},
                         {"device_id": h.OP15, "reason": "KERNEL_RELEASE_MISMATCH", "observed": STOCK_KERNEL,
                          "pinned": PINNED_KERNEL})
        self.assertNotIn("readmit_device", [call[0] for call in self.scheduler.calls])
        self.assertEqual(self.scheduler.quarantined, {h.OP15: "HELPER_LOST"})
        self.assertEqual(self.rig._membership[h.OP15].readmissions, 0)
        # every check ran `uname -r` on the pinned serial, stdin closed
        self.assertEqual(self.log.read_text().splitlines(),
                         [f"-P 5037 -s {h.OP15_SERIAL} shell uname -r stdin=/dev/null"] * 2)
        # the phone boots the pinned kernel again: readmitted, the kernel in its identity
        self.release.write_text(PINNED_KERNEL + "\n")
        self.assertEqual(self.probe(), "MEMBER")
        (_kind, device, identity) = self.scheduler.calls[-1]
        self.assertEqual(identity, canonical_sha256({
            "device_id": h.OP15, "kernel_release": PINNED_KERNEL, "product_id": "4ee7",
            "serial": h.OP15_SERIAL, "vendor_id": "18d1"}))
        self.assertEqual((device, self.scheduler.quarantined), (h.OP15, {}))

    def test_an_unreadable_kernel_release_is_refused(self):
        self.release.write_text("")
        self.assertEqual(self.probe(), "QUARANTINED")
        (rejected,) = self.rejections()
        self.assertEqual((rejected["reason"], rejected["observed"], rejected["pinned"]),
                         ("KERNEL_RELEASE_UNVERIFIED", "error: device offline", PINNED_KERNEL))
        self.assertNotIn("readmit_device", [call[0] for call in self.scheduler.calls])

    def test_refused_joins_do_not_consume_the_readmission_cap(self):
        self.rig._elastic_phones = elastic_phones_configuration({
            "join": True, "join_probe_interval_s": 1, "readmission_cooldown_s": 0,
            "max_readmissions_per_device": 1})
        self.release.write_text(STOCK_KERNEL + "\n")
        for _ in range(3):
            self.assertEqual(self.probe(), "QUARANTINED")
        self.release.write_text(PINNED_KERNEL + "\n")
        self.assertEqual(self.probe(), "MEMBER")
        self.scheduler.quarantined[h.OP15] = "HELPER_LOST"
        self.assertEqual(self.probe(), "QUARANTINED")
        self.assertIn("READMISSION_LIMIT", [row["kind"] for row in self.rig.helper_membership_events])


# ---- review fixes: USB helpers wait for a relaunch, RESET lines are not evidence ------------------

class ResetLineEvidenceTests(unittest.TestCase):
    """``S41SERVERFFNRESET`` lines (a session the server closed after a loss it already reported, and
    the shutdown summary that keeps the last error, LIBUSB text included) never name a helper."""

    ROWS = (("op15", h.OP15, "functionfs-usb"), ("pixel", h.PIXEL, "tcp"))

    def test_the_classifier_skips_reset_lines(self):
        from research_dev.scheduler.adapters.heterogeneous_rig_ops.observations import classify_helper_failure

        resets = ("S41SERVERFFNRESET helper=op15 cause=failed resets=1",
                  "S41SERVERFFNRESET helper=op15 summary resets=1 last_error=libusb_bulk_transfer: "
                  "LIBUSB_ERROR_NO_DEVICE helper op15: gone",
                  "srv  update_slots: decode() failed: Compute aborted.")
        self.assertEqual(classify_helper_failure(resets, "Compute aborted.", self.ROWS), ((), ""))
        # a fresh failure is still named, by its own line
        fresh = (*resets, "S41SERVERFFNERROR helper=pixel detail=FFN split EXECUTE header exchange failed")
        self.assertEqual(classify_helper_failure(fresh, "Compute aborted.", self.ROWS)[0], (h.PIXEL,))
        usb = (*resets, "S41SERVERFFNERROR helper=op15 detail=libusb_bulk_transfer: LIBUSB_ERROR_NO_DEVICE")
        self.assertEqual(classify_helper_failure(usb, None, self.ROWS)[0], (h.OP15,))

    def test_a_later_failure_window_with_reset_lines_does_not_name_the_healthy_op15(self):
        """M2: after the op15 re-joined, a later server-side failure whose stderr window holds the old
        session's reset lines is not an op15 loss (both phones alive: no helper is named)."""
        with h.TemporaryModel() as model, tempfile.TemporaryDirectory() as directory:
            rig = g1d.partial_rig(Path(directory), model, time.monotonic_ns())
            rig.configuration.elastic_phones = MASK_OUT
            rig._failure_evidence_timeout_s = 0.5
            rig._co_helper_sessions = {h.PIXEL: SimpleNamespace(alive=lambda: True)}
            server = masked_server()
            g1d.publish(rig, model, server)
            rig._execution_markers["t0:attempt:0"] = (
                server, None, SimpleNamespace(stderr_index=len(server.lines)), (8, 30))
            server.lines += [
                "S41SERVERFFNRESET helper=op15 cause=failed resets=1",
                "S41SERVERFFNRESET helper=op15 summary resets=1 last_error=LIBUSB_ERROR_NO_DEVICE",
                "srv  update_slots: decode() failed: Compute aborted.",
            ]
            classification = rig._classify_execution_failure(
                desktop_command("t0:attempt:0"), cut(), started_ns=time.monotonic_ns())
        self.assertIsNone(classification)
        self.assertEqual(rig.helper_mask_events, ())


class UsbHelperPendingRelaunchTests(unittest.TestCase):
    """M1: the S2a server never reconnects a FunctionFS helper on the live process (the worker reads
    a new HELLO only after its session was relaunched), so a re-joined OP15 stays masked on that
    server until a planned transition relaunches it, while a TCP helper re-attaches live."""

    def setUp(self):
        scope = h.TemporaryModel()
        self.model = scope.__enter__()
        self.addCleanup(scope.__exit__)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.rig = g1d.partial_rig(Path(directory.name), self.model, time.monotonic_ns())
        self.rig.configuration.elastic_phones = MASK_OUT
        self.rig._failure_evidence_timeout_s = 5.0
        self.server = masked_server()
        g1d.publish(self.rig, self.model, self.server)

    def lose(self, label: str, ticket: str) -> None:
        self.rig._execution_markers[ticket] = (
            self.server, None, SimpleNamespace(stderr_index=len(self.server.lines)), (8, 30))
        self.server.lose_helper(label)
        classification = self.rig._classify_execution_failure(
            desktop_command(ticket), cut(), started_ns=time.monotonic_ns())
        self.assertEqual(classification.masked_executor_id, DESKTOP)

    def test_a_rejoined_usb_helper_waits_for_the_relaunch_and_a_tcp_helper_reattaches_live(self):
        from test_per_device_policies import device_policy

        self.lose("op15", "a:attempt:0")
        self.lose("pixel", "b:attempt:0")
        self.assertEqual(self.rig.masked_helper_devices(DESKTOP), (h.OP15, h.PIXEL))
        self.rig._reattach_masked_helpers(h.OP15)
        self.rig._reattach_masked_helpers(h.PIXEL)
        rows = [row for row in self.rig.helper_mask_events if row["kind"].startswith("HELPER_REATTACH")]
        self.assertEqual([(row["kind"], row["device"], row.get("reason"), row.get("mode")) for row in rows],
                         [("HELPER_REATTACH_PENDING_RELAUNCH", h.OP15, "USB_SESSION", None),
                          ("HELPER_REATTACHED", h.PIXEL, None, "live_reconnect")])
        command = desktop_command("r:attempt:0")
        pixel = device_policy(((h.PIXEL, h.PIXEL_MASK),))
        self.rig._helper_mask_guard(command, control(pixel))
        for owners in (((h.OP15, OP15_LAYERS),), ((h.OP15, OP15_LAYERS), (h.PIXEL, h.PIXEL_MASK))):
            with self.assertRaisesRegex(PhysicalAdapterError, "masked-out or quarantined helper " + h.OP15):
                self.rig._helper_mask_guard(command, control(device_policy(owners)))
        # the planned relaunch of the server (a model switch) ends the mask: the new process connects
        # the OP15 afresh once the transition path relaunched its phone session
        self.rig._stop_executor(DESKTOP, terminate_phone_session=False)
        ended = self.rig.helper_mask_events[-1]
        self.assertEqual((ended["kind"], ended["device"], ended["pending_relaunch"]),
                         ("HELPER_MASK_ENDED", h.OP15, True))
        g1d.publish(self.rig, self.model, masked_server(2))
        self.rig._helper_mask_guard(desktop_command("r2:attempt:0"),
                                    control(device_policy(((h.OP15, OP15_LAYERS), (h.PIXEL, h.PIXEL_MASK)))))

    def test_an_older_capability_line_never_reconnects_live(self):
        """The first S2a build announced ``helper_reconnect=1`` for every transport: not trusted."""
        self.server = masked_server(capabilities="S41SERVERFFNCAPS helper_mask_out=1 helper_reconnect=1")
        g1d.publish(self.rig, self.model, self.server)
        self.lose("pixel", "b:attempt:0")
        self.rig._reattach_masked_helpers(h.PIXEL)
        self.assertEqual(self.rig.helper_mask_events[-1]["kind"], "HELPER_REATTACH_PENDING_RELAUNCH")


class PrimarySessionRelaunchTests(unittest.TestCase):
    """M1: after the primary's loss (mask_out) the next transition that needs its FunctionFS
    session relaunches it (abort the old one, start a new one) instead of reusing it."""

    class Session:
        def __init__(self):
            self.active, self.calls = True, []

        def supports(self, *_args):
            return True

        def supports_partial_reconfiguration(self, *_args):
            return True

        def start(self, *_args, **_options):
            self.calls.append("start")
            self.active = True

        def bind(self, *_args):
            self.calls.append("bind")

    def rig(self, elastic):
        from research_dev.scheduler.adapters.heterogeneous_rig import HeterogeneousPhysicalRig

        rig = HeterogeneousPhysicalRig.__new__(HeterogeneousPhysicalRig)
        rig.configuration = SimpleNamespace(phone_device_id=h.OP15, elastic_phones=elastic)
        rig.epoch_ns = time.monotonic_ns()
        rig._lock = threading.RLock()
        rig._live_executors = {}
        rig._direct_phone_session = self.Session()
        rig._current_direct_phone = rig._direct_phone_session
        rig._direct_phone_receipts = []
        rig._residency_resources = rig._phone_residency_resources = lambda _executor: ((), (), ())
        rig._begin_transition_phone_activity = lambda *_args: None
        rig._direct_phone_failure_resources = lambda *_args: ()
        rig.stops = []
        rig._abort_direct_phone = lambda session: rig.stops.append("abort")
        rig._finish_direct_phone = lambda session, **_options: rig.stops.append("finish")
        return rig

    def transition(self, rig):
        from unittest import mock
        from research_dev.scheduler.adapters.heterogeneous_rig_ops import transitions

        command = SimpleNamespace(
            participant=SimpleNamespace(executor_id=DESKTOP), helper_only=True, ticket_id="t:attempt:1",
            adapter_parameters={"phone_device_id": h.OP15},
            transition=SimpleNamespace(changed_phone_session_ids=()))
        with mock.patch.object(transitions, "phone_transport_contract",
                               return_value=SimpleNamespace(transport="functionfs-usb")):
            state = rig._begin_transition_execution(command, self.model, time.monotonic_ns())
            rig._prepare_functionfs_phone(command, state, lambda: None)
        return state

    def setUp(self):
        scope = h.TemporaryModel()
        self.model = scope.__enter__()
        self.addCleanup(scope.__exit__)

    def test_a_lost_primary_session_is_relaunched_not_reused(self):
        rig = self.rig(MASK_OUT)
        rig._require_direct_phone_relaunch(h.OP15, desktop_command("lost:attempt:0"))
        state = self.transition(rig)
        self.assertFalse(state.direct_phone_reused)
        self.assertEqual((rig.stops, rig._direct_phone_session.calls), (["abort"], ["start"]))
        self.assertFalse(rig._direct_phone_relaunch_pending())
        self.assertEqual([row["kind"] for row in rig.helper_mask_events],
                         ["PRIMARY_SESSION_RELAUNCH_REQUIRED", "PRIMARY_SESSION_RELAUNCHED"])
        # the relaunched session is reused as before
        state = self.transition(rig)
        self.assertTrue(state.direct_phone_reused)
        self.assertEqual(rig._direct_phone_session.calls, ["start", "bind"])

    def test_retire_mode_and_a_co_helper_loss_keep_todays_reuse(self):
        for elastic in (g1d.ELASTIC, MASK_OUT):
            with self.subTest(elastic=dict(elastic)):
                rig = self.rig(elastic)
                rig._helper_lost_classification(desktop_command("t:attempt:0"), (h.PIXEL,), None)
                if elastic is g1d.ELASTIC:
                    rig._helper_lost_classification(desktop_command("u:attempt:0"), (h.OP15,), None)
                self.assertFalse(rig._direct_phone_relaunch_pending())
                self.assertTrue(self.transition(rig).direct_phone_reused)
                self.assertEqual((rig.stops, rig._direct_phone_session.calls), ([], ["bind"]))


if __name__ == "__main__":
    unittest.main()
