"""Per-device adaptive phone policies: the device set (primary alone, primary + co-helper, co-helper
alone) is chosen per batch composition from measured evidence.

The synthetic server has one primary phone ("op15", layers 2-3) and one co-helper ("pixel", layers
4-5). Energies and per-token latencies are given per (device set, batch composition); every window
lasts two tokens. The latency bound is the campaign's 1.25x and the minimum saving 1 %."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from research_dev.scheduler import UnifiedScheduler
from research_dev.scheduler._internal.adaptive_decode import AdaptiveDecodeController
from research_dev.scheduler._internal.adaptive_decode_contracts import (
    AdaptiveDecodeConfig, AdaptiveDecodePolicyAck, AdaptiveDecodeRawWindowObservation,
)
from research_dev.scheduler._internal.adaptive_decode_ops.coherence import server_verdict
from research_dev.scheduler._internal.adaptive_decode_planning import (
    adaptive_decode_policies, adaptive_probe_contracts,
)
from research_dev.scheduler._internal.plan_contracts.phone import RuntimePhoneShard
from research_dev.scheduler._internal.types import canonical_sha256
from research_dev.scheduler.adapters.contracts import PhysicalAdapterError
from research_dev.scheduler.adapters.http_backend import _AdaptivePayloadController
from research_dev.scheduler.adapters.llama_server_contracts import LlamaServerFfnCall
from research_dev.scheduler.adapters.llama_server_ops.proofs import ManagedServerProofMixin

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
import two_phone_harness as h  # noqa: E402
from test_adaptive_decode import ARTIFACT, PLAN, policy  # noqa: E402

PRIMARY, CO_HELPER = "op15", "pixel"
PRIMARY_MASK, CO_HELPER_MASK = 0b001100, 0b110000
HOST, P, PC, C = (), (PRIMARY,), (PRIMARY, CO_HELPER), (CO_HELPER,)


def devices(row) -> tuple[str, ...]:
    # local on purpose: the module imports on a tree without per-device policies
    return tuple(device for device, _mask in row.device_layer_masks)


def device_policy(owners, columns=1000):
    layers = tuple(index for index in range(64)
                   if sum(mask for _device, mask in owners) >> index & 1)
    name = "phone-" + "+".join(device for device, _mask in owners) + "-" + str(columns)
    return replace(policy(name, columns, layers), device_layer_masks=tuple(owners))


class _Server:
    """Drives a coherent server: every tick records one two-token window per live session, all
    ending at the same time, and acknowledges every control at once."""

    def __init__(self, test, energies, latencies):
        self.test, self.energies, self.latencies = test, energies, latencies
        self.controller = AdaptiveDecodeController()
        self.now = 1_000
        self.slots = {}
        self.controls = []

    def start(self, request_id, slot_id, *, active_batch=1, output_tokens=400):
        self.slots[request_id] = slot_id
        directive = self.controller.start(
            request_id=request_id, ticket_id=request_id + ":attempt:0",
            model_artifact_sha256=ARTIFACT, planning_profile_sha256=PLAN,
            baseline=self.test.baseline, candidates=self.test.candidates,
            output_tokens=output_tokens, context_length=64, active_batch=active_batch,
            deadline_us=10**12, slot_id=slot_id, first_token_index=1, first_token_at_us=self.now,
            config=self.test.config, helper_evidence_state="LEARNING",
            helper_layout_generation=1, helper_layout_geometry_sha256="sha256:" + "7" * 64)
        self._acknowledge(request_id, directive)

    def _acknowledge(self, request_id, directive):
        while directive.control is not None:
            self.controls.append((request_id, devices(directive.control.policy)))
            session = self.controller._sessions[request_id]
            directive = self.controller.acknowledge(request_id, AdaptiveDecodePolicyAck(
                request_id, self.slots[request_id], directive.control.plan_generation,
                session.transition_start_token, self.now, directive.control.policy.policy_hash))
        return directive

    def tick(self, *, next_active_batch=None, failure=None, only=None):
        sessions = [(request_id, self.controller._sessions[request_id]) for request_id in self.slots
                    if only is None or request_id in only]
        duration = max(self.latencies[(devices(session.current_policy), session.active_batch)]
                       for _request_id, session in sessions) * 2
        self.now += duration
        results = {}
        for request_id, session in sessions:
            running = devices(session.current_policy)
            phone = not session.current_policy.baseline
            energy = self.energies[(running, session.active_batch)]
            boundary = self.controller.boundary(request_id, slot_id=self.slots[request_id],
                                                token_index=session.target_token, at_us=self.now).boundary
            batch = None if next_active_batch is None else next_active_batch.get(request_id)
            directive = self.controller.record_window(request_id, boundary, AdaptiveDecodeRawWindowObservation(
                fleet_energy_uj_by_domain={"fleet": energy * boundary.token_count},
                phone_compute_us=0, usb_transfer_us=0, rpc_us=0, exposed_tail_us=0, output_valid=True,
                evidence_ids=("synthetic:window",), energy_boundary_id="synthetic-fleet",
                energy_attribution_kind="isolated",
                failure_reason=failure if failure is not None and phone else None,
                usb_upload_bytes=0, usb_download_bytes=0, desktop_compute_us=20, useful_overlap_us=0,
                request_queue_delay_us=3, protected_interference_us=0, active_batch=None,
                next_active_batch=batch, membership_changed=batch is not None,
                execution_context_available=True, completed_phone_calls=2 if phone else 0,
                completed_phone_input_rows=2 if phone else 0, external_activity_sha256=None),
                **({} if batch is None else {"compatible_batch_change": True}))
            results[request_id] = self._acknowledge(request_id, directive)
        return results

    def running(self, request_id="request-a"):
        return devices(self.controller.active_policy(request_id))

    def group(self, request_id="request-a"):
        return self.controller._server_policies[self.controller.shared_server_policy_key(request_id)]


class _DeviceSetCase(unittest.TestCase):
    def setUp(self):
        self.baseline = policy("desktop-control", 0, (), baseline=True)
        self.primary = device_policy(((PRIMARY, PRIMARY_MASK),))
        self.both = device_policy(((PRIMARY, PRIMARY_MASK), (CO_HELPER, CO_HELPER_MASK)))
        self.helper = device_policy(((CO_HELPER, CO_HELPER_MASK),))
        self.candidates = (self.primary, self.both, self.helper)
        self.config = AdaptiveDecodeConfig(
            minimum_remaining_tokens=4, minimum_window_tokens=2, maximum_window_tokens=2,
            maximum_probe_tokens=20, maximum_probe_candidates=1, measurement_resolution_us=1,
            transition_cost_us=1, transition_energy_uj=1, minimum_energy_saving_ppm=10_000,
            uncertainty_ppm=10_000, warmup_windows_per_policy=0, maximum_latency_ppm=1_250_000,
            maximum_probe_attempts_per_context=2, server_policy_coherence=True)
        # batch 1: the co-helper helps; batch 2: it saves energy but breaks the latency bound
        self.energies = {(HOST, 1): 100, (P, 1): 70, (PC, 1): 45, (C, 1): 80,
                         (HOST, 2): 60, (P, 2): 40, (PC, 2): 35, (C, 2): 55}
        self.latencies = {(HOST, 1): 1_000, (P, 1): 1_000, (PC, 1): 1_100, (C, 1): 1_200,
                          (HOST, 2): 1_000, (P, 2): 1_050, (PC, 2): 3_000, (C, 2): 1_300}

    def server(self):
        return _Server(self, self.energies, self.latencies)

    def qualify_both_at_batch_one(self, server):
        """Cold start: host, the primary alone (verdict), then the co-helper added (verdict)."""
        server.start("request-a", 1)
        server.tick()
        self.assertEqual(server.running(), P)
        server.tick()
        self.assertEqual(server_verdict(server.group(), 1), self.primary)
        server.tick()
        self.assertEqual(server.running(), PC)
        server.tick()
        self.assertEqual(server_verdict(server.group(), 1), self.both)
        return server.group()


class DeviceSetPlanningTests(unittest.TestCase):
    """The policy space of a two-phone envelope: every device set on the envelope's column grid."""

    def setUp(self):
        scope = h.TemporaryModel()
        self.model = scope.__enter__()
        self.addCleanup(scope.__exit__)

    def _policies(self, catalog):
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(catalog)
        scheduler.register_model_manifest(self.model)
        values = scheduler.generate_automated_candidates(
            h.request("per-device"), self.model.model_id, h.snapshot(self.model, catalog))
        return values, adaptive_decode_policies(values, self.model, catalog, 30)

    def test_two_phone_envelope_yields_every_device_set_on_the_grid(self):
        catalog = h.runtime_catalog(self.model, declaration=h.co_helpers())
        values, (_baseline, policies, envelope) = self._policies(catalog)
        self.assertIsNotNone(envelope)
        by_set = {}
        for row in policies:
            by_set.setdefault(devices(row), []).append(row)
        self.assertEqual(set(by_set), {(h.OP15, h.PIXEL), (h.OP15,), (h.PIXEL,)})
        for owners, rows in by_set.items():
            with self.subTest(devices=owners):
                self.assertEqual(sorted(row.columns for row in rows), [32, 64, 96, 128])
                mask = sum(dict(rows[0].device_layer_masks).values())
                self.assertTrue(all(row.layer_mask == mask for row in rows))
        self.assertEqual(sum(dict(by_set[(h.OP15,)][0].device_layer_masks).values()), 0b1111)
        self.assertEqual(sum(dict(by_set[(h.PIXEL,)][0].device_layer_masks).values()), h.PIXEL_MASK)
        # the envelope's probe contract carries them, so the ticket admits their controls
        contract = adaptive_probe_contracts(values, self.model, catalog, 30)[envelope.candidate_id]
        probed = {tuple(device for device, _mask in row.get("device_layer_masks", ()))
                  for row in contract["adaptive_decode_probe_contracts"]}
        self.assertEqual(probed, set(by_set))

    def test_single_phone_envelope_has_no_device_subsets(self):
        _values, (_baseline, policies, envelope) = self._policies(h.runtime_catalog(self.model))
        self.assertIsNotNone(envelope)
        self.assertTrue(policies)
        self.assertTrue(all(row.device_layer_masks == () for row in policies))
        self.assertFalse(any(":devices:" in row.route_id for row in policies))


class DeviceSetColdStartTests(_DeviceSetCase):
    def test_cold_start_probes_the_primary_alone_first_then_adds_the_co_helper(self):
        rows = []
        for columns in (1000, 750, 500, 250):
            rows.extend((device_policy(((PRIMARY, PRIMARY_MASK),), columns),
                         device_policy(((PRIMARY, PRIMARY_MASK), (CO_HELPER, CO_HELPER_MASK)), columns),
                         device_policy(((CO_HELPER, CO_HELPER_MASK),), columns)))
        sampled = AdaptiveDecodeController._sample_candidates(
            rows, replace(self.config, maximum_probe_candidates=4))
        self.assertEqual([(devices(row), row.split_fraction_ppm) for row in sampled],
                         [(P, 1_000_000), (PC, 1_000_000), (P, 750_000), (P, 500_000)])
        self.assertEqual([devices(row) for row in AdaptiveDecodeController._sample_candidates(
            rows, self.config)], [P])

    def test_first_server_proposal_is_the_primary_alone(self):
        server = self.server()
        server.start("request-a", 1)
        server.tick()
        self.assertEqual(server.controls, [("request-a", P)])
        self.assertEqual(server.group().proposal, self.primary)
        self.assertEqual(server.controller.snapshot("request-a")["device_sets"],
                         ("op15", "op15+pixel", "pixel"))


class DeviceSetServerVerdictTests(_DeviceSetCase):
    def test_co_helper_that_improves_replaces_the_primary_at_that_batch(self):
        server = self.server()
        group = self.qualify_both_at_batch_one(server)
        self.assertEqual(server.controls, [("request-a", P), ("request-a", PC)])
        self.assertIsNone(group.proposal)
        snapshot = server.controller.snapshot("request-a")["server_policy"]
        self.assertEqual(snapshot["verdict_device_sets"], {"1": "op15+pixel"})
        self.assertEqual(snapshot["device_set_drops"], [])
        # evidence stays per device set: three policies, three identities, each with its windows
        records = server.controller._sessions["request-a"].records
        self.assertEqual({devices(row.policy) for row in records}, {HOST, P, PC})
        server.tick()
        self.assertEqual(server.running(), PC)

    def test_slower_co_helper_is_dropped_for_that_batch_composition_only(self):
        server = self.server()
        group = self.qualify_both_at_batch_one(server)
        server.start("request-b", 2, active_batch=2)
        server.tick(next_active_batch={"request-a": 2})
        # the new composition starts from the cheapest device set, the primary alone
        self.assertEqual(server.running(), P)
        for _ in range(8):
            server.tick()
        self.assertEqual(server_verdict(group, 2), self.primary)
        self.assertEqual(group.device_drops, ((2, PC, "SERVER_DEVICE_SET_LATENCY_BOUND_EXCEEDED"),))
        self.assertEqual(server_verdict(group, 1), self.both)
        self.assertEqual((server.running("request-a"), server.running("request-b")), (P, P))
        # the partner leaves: back at batch 1 the co-helper verdict of that composition applies
        server.controller.complete("request-b", "COMPLETED")
        del server.slots["request-b"]
        server.tick(next_active_batch={"request-a": 1})
        server.tick()
        self.assertEqual(server.running(), PC)
        self.assertEqual(server_verdict(group, 1), self.both)

    def test_co_helper_that_does_not_improve_is_dropped_and_the_primary_kept(self):
        self.energies[(PC, 1)] = 72
        server = self.server()
        server.start("request-a", 1)
        for _ in range(8):
            server.tick()
        group = server.group()
        self.assertEqual(server_verdict(group, 1), self.primary)
        self.assertEqual(group.device_drops, ((1, PC, "SERVER_DEVICE_SET_NOT_IMPROVED"),))
        self.assertEqual(server.running(), P)


class DeviceSetDecisionRecordTests(_DeviceSetCase):
    def _decisions(self, server):
        from research_dev.scheduler._unified.adaptive_decode_control import AdaptiveDecodeControlMixin
        events = []
        owner = SimpleNamespace(_adaptive_decode=server.controller,
                                _model_placement_controller=SimpleNamespace(
                                    record_request_helper_event=lambda *row: events.append(row)))
        return owner, events, AdaptiveDecodeControlMixin._record_assistance_decision

    def test_decision_records_carry_the_device_set_and_the_server_choice(self):
        server = self.server()
        self.qualify_both_at_batch_one(server)
        owner, events, record = self._decisions(server)
        directive = server.tick()["request-a"]
        record(owner, "request-a", directive, server.controller._sessions["request-a"].window_start_token,
               server.now)
        (_request, kind, _at, values), = events
        self.assertEqual(kind, "ASSISTANCE_DECISION")
        self.assertEqual((values["device_set"], values["active_batch"]), ([PRIMARY, CO_HELPER], 1))
        self.assertEqual(values["device_layer_masks"], [[PRIMARY, PRIMARY_MASK], [CO_HELPER, CO_HELPER_MASK]])
        self.assertEqual(values["server_policy"]["verdict_device_sets"], {"1": "op15+pixel"})
        self.assertEqual(values["server_policy"]["device_sets"], ["op15", "op15+pixel", "pixel"])

    def test_single_phone_decision_records_are_unchanged(self):
        self.candidates = (policy("phone-full", 1000, (2, 3)),)
        server = _Server(self, {((), 1): 100}, {((), 1): 1_000})
        server.start("request-a", 1)
        owner, events, record = self._decisions(server)
        directive = server.tick()["request-a"]
        record(owner, "request-a", directive, 3, server.now)
        (_request, _kind, _at, values), = events
        self.assertFalse({"device_set", "device_layer_masks", "active_batch"} & set(values))
        self.assertFalse({"device_sets", "verdict_device_sets"} & set(values["server_policy"] or {}))


class DeviceSetFailureTests(_DeviceSetCase):
    def test_failures_fall_back_to_the_remaining_device_then_the_host(self):
        server = self.server()
        group = self.qualify_both_at_batch_one(server)
        # the two-phone policy fails: that set (and its supersets) is dropped for every composition,
        # the server recovers on the host and falls back to the primary alone, whose measured pair
        # at this composition still qualifies it
        del server.controls[:]
        server.tick(failure="helper pixel: timeout")
        self.assertEqual(group.device_drops, ((0, PC, "SERVER_PHONE_POLICY_FAILED"),))
        self.assertEqual(server.controls, [("request-a", HOST), ("request-a", P)])
        self.assertEqual(server_verdict(group, 1), self.primary)
        self.assertEqual(server.running(), P)
        server.tick()
        self.assertEqual(server.running(), P)
        # the primary fails too: only then the co-helper alone is probed
        server.tick(failure="helper op15: timeout")
        self.assertEqual({row[1] for row in group.device_drops}, {P, PC})
        for _ in range(3):
            server.tick()
        self.assertEqual(server_verdict(group, 1), self.helper)
        self.assertEqual(server.running(), C)
        # the last phone fails: the host is the verdict
        server.tick(failure="helper pixel: timeout")
        self.assertEqual(server_verdict(group, 1), self.baseline)
        self.assertEqual(server.running(), HOST)
        self.assertEqual(server.controller.snapshot("request-a")["server_policy"]["reason"],
                         "SERVER_PHONE_POLICY_FAILED")

    def test_primary_session_drain_leaves_a_co_helper_only_policy_running(self):
        """A drain narrows the primary phone's sessions: the co-helper alone keeps running, the
        two-phone policy keeps only the primary's retained layers (Stage A behaviour)."""
        server = self.server()
        self.qualify_both_at_batch_one(server)
        drain = server.controller.request_helper_session_drain("request-a", retained_layer_mask=0b0100)
        self.assertEqual(drain["retained_layer_mask"], 0b0100)
        pending = server.controller._sessions["request-a"].pending_session_drain_policy
        self.assertEqual((pending.layer_mask, pending.device_layer_masks), (0b0100, ((PRIMARY, 0b0100),)))
        server = self.server()
        server.start("request-a", 1)
        group = server.group()
        for failed in (P, PC):
            group.device_drops += ((0, failed, "SERVER_PHONE_POLICY_FAILED"),)
        for _ in range(4):
            server.tick()
        self.assertEqual(server.running(), C)
        drain = server.controller.request_helper_session_drain("request-a", retained_layer_mask=PRIMARY_MASK)
        self.assertEqual((drain["already_applied"], drain["policy_hash"]), (True, self.helper.policy_hash))
        self.assertIsNone(server.controller._sessions["request-a"].pending_session_drain_policy)

    def test_request_local_controller_drops_the_failed_set_and_its_supersets(self):
        server = self.server()
        self.config = replace(self.config, server_policy_coherence=False, maximum_probe_candidates=4)
        server.start("request-a", 1)
        session = server.controller._sessions["request-a"]
        from research_dev.scheduler._internal.adaptive_decode_ops.coherence import server_policy_failed
        server_policy_failed(server.controller, session, server.now, self.primary)
        self.assertEqual(set(session.eliminated_policy_reasons),
                         {self.primary.policy_hash, self.both.policy_hash})


class PerDeviceProofTests(unittest.TestCase):
    """A phone the executed device sets never drove need not serve calls; one they drove must."""

    def setUp(self):
        scope = h.TemporaryModel()
        self.model = scope.__enter__()
        self.addCleanup(scope.__exit__)
        self.plan_sha256 = "sha256:" + "4" * 64
        self.op15_shard = RuntimePhoneShard(
            "HTP0", "functionfs://op15/HTP0", 0b1111, 128, 1 << 20, "sha256:" + "5" * 64,
            self.plan_sha256, self.model.artifact_sha256, 3)

    def _proofs(self, calls, layer_masks, *, with_helper=True):
        parameters = {
            "phone_co_helpers_v1": h.co_helpers_json(),
            "phone_helpers": h.phone_helpers_json(0b1111),
        } if with_helper else {}
        source = SimpleNamespace(
            operator_plan_sha256=self.plan_sha256, adapter_parameters=parameters,
            execution_contract=SimpleNamespace(remote_resident_ffn=None, phone_shards=(self.op15_shard,)))
        windows = tuple(SimpleNamespace(policy=SimpleNamespace(
            baseline=False, operator_plan_sha256=self.plan_sha256, layer_mask=mask)) for mask in layer_masks)
        observation = SimpleNamespace(windows=windows, unmeasured_tail_tokens=0,
                                      final_policy=SimpleNamespace(baseline=True))
        return ManagedServerProofMixin()._execution_session_proofs(
            SimpleNamespace(artifact_sha256=self.model.artifact_sha256), source, (), observation,
            calls, {}, self.model, set())

    @staticmethod
    def _call(request_id, layer):
        return LlamaServerFfnCall(request_id=request_id, layer=layer, tokens=1, columns=32, payload_bytes=64)

    def test_primary_only_request_needs_no_co_helper_calls(self):
        proofs = self._proofs((self._call(1, 0), self._call(2, 3)), (0b1111,))
        self.assertEqual([(row.session_id, row.calls) for row in proofs], [("HTP0", 2)])

    def test_co_helper_only_request_needs_no_primary_calls(self):
        proofs = self._proofs((self._call(1 + (1 << 24), 4), self._call(2 + (1 << 24), 5)), (h.PIXEL_MASK,))
        self.assertEqual([(row.session_id, row.calls) for row in proofs], [("PIXEL0", 2)])

    def test_every_phone_of_an_executed_set_must_serve_calls(self):
        with self.assertRaises(PhysicalAdapterError):
            self._proofs((self._call(1, 0), self._call(2, 3)), (0b1111, 0b1111 | h.PIXEL_MASK))
        proofs = self._proofs((self._call(1, 0), self._call(1 + (1 << 24), 4)), (0b1111, 0b1111 | h.PIXEL_MASK))
        self.assertEqual(sorted((row.session_id, row.calls) for row in proofs), [("HTP0", 1), ("PIXEL0", 1)])

    def test_single_phone_proofs_still_require_every_shard(self):
        second = replace(self.op15_shard, session_id="HTP1", endpoint="functionfs://op15/HTP1",
                         layer_mask=0b110000)
        source = SimpleNamespace(operator_plan_sha256=self.plan_sha256, adapter_parameters={},
                                 execution_contract=SimpleNamespace(
                                     remote_resident_ffn=None, phone_shards=(self.op15_shard, second)))
        observation = SimpleNamespace(
            windows=(SimpleNamespace(policy=SimpleNamespace(
                baseline=False, operator_plan_sha256=self.plan_sha256, layer_mask=0b1111)),),
            unmeasured_tail_tokens=0, final_policy=SimpleNamespace(baseline=True))
        with self.assertRaises(PhysicalAdapterError):
            ManagedServerProofMixin()._execution_session_proofs(
                SimpleNamespace(artifact_sha256=self.model.artifact_sha256), source, (), observation,
                (self._call(1, 0),), {}, self.model, set())


class PerDeviceEnergyTests(unittest.TestCase):
    def _controller(self):
        primary, helper = [], []
        controller = _AdaptivePayloadController.__new__(_AdaptivePayloadController)
        controller.backend = SimpleNamespace(_epoch_ns=1000, _energy_meter=SimpleNamespace(
            record_phone_activity_duration=lambda *row: primary.append(row),
            record_helper_phone_window=lambda *row: helper.append(row)))
        controller.command = SimpleNamespace(adapter_parameters={
            "phone_helpers": h.phone_helpers_json(0b1111), "phone_device_id": h.OP15})
        return controller, primary, helper

    def test_co_helper_only_window_leaves_the_primary_idle(self):
        controller, primary, helper = self._controller()
        delta = {"rpc_us": 5, "usb_transfer_us": 1, "phone_compute_us": 4}
        controller._record_phone_window("qwen", 10, 20, delta,
                                        SimpleNamespace(device_layer_masks=((h.PIXEL, h.PIXEL_MASK),)))
        self.assertEqual(primary, [])
        self.assertEqual(helper[0][-1], (h.PIXEL,))
        controller._record_phone_window("qwen", 20, 30, delta,
                                        SimpleNamespace(device_layer_masks=((h.OP15, 0b1111),)))
        self.assertEqual(primary[0][-1], 5)
        self.assertEqual(helper[1][-1], (h.OP15,))


class SinglePhoneDigestGuardTests(_DeviceSetCase):
    """One helper phone: candidate sampling, the coherent server trace and its snapshots are those
    of the base tree (digests computed on the tree before per-device policies)."""

    GOLDEN = "sha256:e4eaec0fb19d5a7c239e429240a873f5cb5f7fb85f70d383698dc7c61d7373c5"

    def test_single_phone_server_trace_matches_the_base_tree(self):
        """Probe, verdict, a co-tenant at batch 2 with its like-for-like comparison, the partner
        leaving, then a failed phone window: every snapshot and control as on the base tree."""
        rows = [policy("phone-" + str(columns), columns, (2, 3)) for columns in (250, 500, 750, 1000)]
        sampled = AdaptiveDecodeController._sample_candidates(
            rows, replace(self.config, maximum_probe_candidates=4))
        self.candidates = tuple(rows)
        phone_energy = {(1000, 1): 45, (750, 1): 50, (500, 1): 60, (250, 1): 80, (1000, 2): 38}
        host_energy = {1: 100, 2: 60}
        server = _Server(self, {}, {((), 1): 1_000, ((), 2): 1_000})
        trace = []

        def tick(**arguments):
            # one phone: every policy has the empty device set, so energy follows width and batch
            for request_id in server.slots:
                session = server.controller._sessions[request_id]
                current, batch = session.current_policy, session.active_batch
                server.energies[((), batch)] = (host_energy[batch] if current.baseline
                                                else phone_energy[(current.columns, batch)])
            server.tick(**arguments)
            for request_id in sorted(server.slots):
                snapshot = server.controller.snapshot(request_id)
                self.assertNotIn("device_sets", snapshot)
                trace.append({key: snapshot[key] for key in (
                    "server_policy", "stage", "state", "zero_assistance_reason", "current_policy_hash",
                    "eliminated_policy_reasons", "active_batch")})

        server.start("request-a", 1)
        for _ in range(6):
            tick()
        server.start("request-b", 2, active_batch=2)
        tick(next_active_batch={"request-a": 2})
        for _ in range(8):
            tick()
        server.controller.complete("request-b", "COMPLETED")
        del server.slots["request-b"]
        tick(next_active_batch={"request-a": 1})
        tick()
        tick(failure="phone: timeout")
        for _ in range(3):
            tick()
        digest = canonical_sha256({"sampled": [row.to_json() for row in sampled], "trace": trace,
                                   "controls": server.controls})
        self.assertEqual(digest, self.GOLDEN)


if __name__ == "__main__":
    unittest.main()
