#!/usr/bin/env python3
"""Joint planner active mode (``dispatch_policy.joint_planner = {"mode": "active", ...}``).

(a) the s2a Qwen 001/003/004 sequence co-decodes under the active mode with shared phone lanes
    (a token-level model of llama-server's slot batching rules driving the real adaptive controller,
    the real helper lease sharing and the real active planner hook);
(b) fallback paths (budget expiry, planner failure, capture failure, refused registration, admission
    not overridden, advisory-only actions, yield expiry);
(c) shadow mode and the absent key are unchanged;
(d) the planner touches no admission, lease, identity or thermal check (decision log bytes of the real
    scheduler identical with the active mode on; a yield only ever issues host policies; every phone
    control after it is the one the unchanged controller issues).
"""

from __future__ import annotations

import ast
from contextlib import nullcontext
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from research_dev.scheduler._internal.adaptive_decode import AdaptiveDecodeController
from research_dev.scheduler._internal.adaptive_decode_contracts import (
    AdaptiveDecodeConfig, AdaptiveDecodePolicyAck, AdaptiveDecodeRawWindowObservation,
)
from research_dev.scheduler._internal.adaptive_decode_ops.prefill_yield import PREFILL_YIELD_REASON
from research_dev.scheduler._internal.joint_planner import PlanResult
from research_dev.scheduler._internal.joint_planner_active import (
    FALLBACK, JOINT_PLANNER_ACTIVE_SCHEMA, JointPlannerActive, JointPlannerActiveConfig,
    JointPlannerActiveError, classify_plan, joint_planner_config_from_json, joint_planner_from_policy_json,
)
from research_dev.scheduler._internal.joint_planner_model import measured_eval_v2_cost_model
from research_dev.scheduler._internal.joint_planner_shadow import (
    JointPlannerShadowConfig, JointPlannerShadowError, ShadowEpoch, simulator_from_epoch,
)
from research_dev.scheduler._internal.joint_planner_sim import Admit, Park, Provision, SimRequest, Switch
from research_dev.scheduler._unified.adaptive_decode_control import AdaptiveDecodeControlMixin
from research_dev.scheduler._unified.helper_preparation_ops.attachment import _attach_helper_window_leases
from research_dev.scheduler._unified.helper_preparation_ops.cleanup import _release_request_helper_leases
from research_dev.scheduler._internal.lifecycle import UnifiedScheduleError
from research_dev.scheduler.campaigns.burstgpt import runner
from research_dev.scheduler.configuration.campaign import _dispatch_policy
from research_dev.scheduler.configuration.common import SchedulerConfigurationError
from research_dev.scheduler.tests.test_adaptive_decode import ARTIFACT, PLACEMENT, PLAN, policy

import test_dispatch_policy as dispatch_harness

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "campaigns" / "burstgpt"))
import joint_planner_eval as evaluation  # noqa: E402
import joint_planner_active_eval as active_evaluation  # noqa: E402


QWEN = "qwen3-14b-q4km-dequant-f16"
GEOMETRY = "sha256:" + "7" * 64
ROOT = Path(__file__).resolve().parents[1]
COST = measured_eval_v2_cost_model()


def adaptive_config(**overrides) -> AdaptiveDecodeConfig:
    values = dict(
        minimum_remaining_tokens=4, minimum_window_tokens=4, maximum_window_tokens=4,
        maximum_probe_tokens=20, maximum_probe_candidates=1, measurement_resolution_us=1,
        transition_cost_us=1, transition_energy_uj=1, minimum_energy_saving_ppm=10_000,
        uncertainty_ppm=10_000, warmup_windows_per_policy=0, server_policy_coherence=True,
        batch_growth_verdict_inheritance=True,
    )
    values.update(overrides)
    return AdaptiveDecodeConfig(**values)


BASELINE = policy("desktop-control", 0, (), baseline=True)
FULL = policy("phone-full", 1000, (2, 3))


def observation(energy_per_token: int, tokens: int, phone: bool, *, next_active_batch=None,
                membership_changed=False) -> AdaptiveDecodeRawWindowObservation:
    return AdaptiveDecodeRawWindowObservation(
        fleet_energy_uj_by_domain={"fleet": energy_per_token * tokens},
        phone_compute_us=0, usb_transfer_us=0, rpc_us=0, exposed_tail_us=0, output_valid=True,
        evidence_ids=("synthetic:window",), energy_boundary_id="synthetic-fleet",
        energy_attribution_kind="isolated", failure_reason=None, usb_upload_bytes=0,
        usb_download_bytes=0, desktop_compute_us=20, useful_overlap_us=0, request_queue_delay_us=3,
        protected_interference_us=0, active_batch=None, next_active_batch=next_active_batch,
        membership_changed=membership_changed, execution_context_available=True,
        completed_phone_calls=tokens if phone else 0, completed_phone_input_rows=tokens if phone else 0,
        external_activity_sha256=None)


# A token-level model of the llama-server rules that decide which slot runs --------------------------


class FakeLlamaServer:
    """``tools/server/server-context.cpp``: slots batch only with equal FFN split policies
    (``can_batch_with`` :407, compared on layer mask and columns :81); a new slot starts on the host
    policy (:333) and an FFN control needs decode state (:2741); the batch anchor rotates among
    decoding slots only (:3248-3261); a prompt joins only a batch it can batch with (:3409). One
    forward decodes one token per batched decoding slot and processes whole prompts; its duration is
    the measured cost model (prompt, then the decode step at the batch's size and policy)."""

    def __init__(self, start_s: float) -> None:
        self.costs = COST.model("qwen")
        self.now = start_s
        self.slots: dict[int, dict] = {}
        self.cursor = 0
        self.forwards: list[dict] = []

    def admit(self, slot_id: int, request_id: str, input_tokens: int, output_tokens: int, *,
              decoded: int = 0, phone: bool = False) -> None:
        self.slots[slot_id] = {"request_id": request_id, "state": "GENERATING" if decoded else "PROMPT",
                               "prompt": input_tokens, "n": decoded, "out": output_tokens, "phone": phone}

    def processing(self) -> int:
        return sum(1 for slot in self.slots.values() if slot["state"] in ("PROMPT", "GENERATING"))

    def control(self, slot_id: int, phone: bool) -> None:
        slot = self.slots[slot_id]
        if slot["state"] != "GENERATING":
            raise AssertionError("FFN control requires decode state")
        slot["phone"] = phone

    def step(self) -> list[tuple[int, str, int, bool]]:
        ids = sorted(self.slots)
        anchor = None
        for offset in range(len(ids)):
            index = ids[(self.cursor + offset) % len(ids)]
            if self.slots[index]["state"] == "GENERATING":
                anchor = index
                self.cursor = (ids.index(index) + 1) % len(ids)
                break
        phone = anchor is not None and self.slots[anchor]["phone"]
        decode = [] if anchor is None else [
            i for i in ids if self.slots[i]["state"] == "GENERATING" and self.slots[i]["phone"] == phone]
        prompt = [i for i in ids if self.slots[i]["state"] == "PROMPT" and not phone]
        if not decode and not prompt:
            return []
        duration = 0.0
        if prompt:
            duration += self.costs.prefill_s(sum(self.slots[i]["prompt"] for i in prompt))
        if decode:
            duration += self.costs.step_s(len(decode), phone)
        self.now += duration
        produced = []
        for i in decode + prompt:
            slot = self.slots[i]
            slot["n"] += 1
            slot["state"] = "DONE" if slot["n"] >= slot["out"] else "GENERATING"
            produced.append((i, slot["request_id"], slot["n"], slot["state"] == "DONE"))
        self.forwards.append({
            "t": self.now, "decode": [self.slots[i]["request_id"] for i in decode],
            "prompt": [self.slots[i]["request_id"] for i in prompt], "phone": phone,
            "phone_rows": len(decode) if phone else 0,
        })
        return produced


class Lanes:
    """The real helper-window lease code (``_attach_helper_window_leases``,
    ``_release_request_helper_leases``) on a stand-in scheduler whose phone lanes have capacity 1."""

    def __init__(self) -> None:
        self.tickets: dict[str, SimpleNamespace] = {}
        self.bindings: dict[str, dict] = {}
        self.live: set[str] = set()
        self.events: list[tuple] = []
        self.counter = 0
        self.helper = SimpleNamespace(
            desktop_placement_sha256=PLACEMENT, phone_layout_generation=1, phone_layout_geometry_sha256=GEOMETRY,
            resident_layer_mask=FULL.layer_mask, resident_columns=1000,
            helper_plan=SimpleNamespace(resource_ids=("cpu", "gpu", "phone", "usb"),
                                        execution_contract=SimpleNamespace(maximum_batch_size=4)),
        )
        self.controller = SimpleNamespace(
            _adaptive_decode_config=AdaptiveDecodeConfig(server_policy_coherence=True),
            _runtime_controller=SimpleNamespace(current_tickets=lambda _states: list(self.tickets.values())),
            _model_placement_controller=SimpleNamespace(
                request_binding=self.bindings.get, record_request_helper_event=self._event,
                update_request_fraction=self._fraction, renew_request_helper_leases=self._renew),
            _helper_window_lease_horizon=lambda *args, **kwargs: 10**13,
            _transaction=lambda **kwargs: nullcontext(), extend_lease=lambda token, until: None,
            reserve_external_resources=self._reserve,
            _select_helper_window_owner=lambda request_id, resources, **kwargs: (request_id, None),
            _runtime_renewals={}, release=self._release,
        )
        self.controller._release_request_helper_leases = (
            lambda request_id, at_us: _release_request_helper_leases(self.controller, request_id, at_us))

    def acquire(self, request_id: str) -> None:
        self.tickets[request_id] = SimpleNamespace(
            request=SimpleNamespace(request_id=request_id), ticket_id=request_id + ":0",
            model=SimpleNamespace(artifact_sha256=ARTIFACT), binding=SimpleNamespace(endpoint="http://desktop:18571"),
            execution_plan=SimpleNamespace(resource_ids=("cpu", "gpu"), adapter_parameters={"parallel": 4}))
        self.bindings[request_id] = {
            "base": {"desktop_placement_sha256": PLACEMENT}, "fraction_ppm": 0,
            "helper_envelope": {"assisted_layer_mask": FULL.layer_mask, "maximum_columns": 1000,
                                "resource_ids": ("cpu", "gpu", "phone", "usb")},
            "helper_attachment": None,
        }

    def complete(self, request_id: str, at_us: int) -> None:
        _release_request_helper_leases(self.controller, request_id, at_us)
        self.tickets.pop(request_id, None)

    def apply(self, request_id: str, phone: bool, token_index: int, at_us: int) -> None:
        binding = self.bindings[request_id]
        if not phone:
            _release_request_helper_leases(self.controller, request_id, at_us)
            return
        leases = _attach_helper_window_leases(
            self.controller, self.tickets[request_id], self.helper, SimpleNamespace(generation=1),
            current_attachment=binding["helper_attachment"], token_index=token_index, at_us=at_us,
            fraction_ppm=1_000_000)
        if not hasattr(leases, "lease_tokens"):
            raise AssertionError("helper window lease refused: " + leases.reason)
        binding["fraction_ppm"] = 1_000_000
        binding["helper_attachment"] = {
            "lease_tokens": tuple(leases.lease_tokens), "lease_reserved_until_us": leases.finish_us,
            "phone_layout_generation": 1, "phone_layout_geometry_sha256": GEOMETRY,
        }

    def tokens(self, request_id: str) -> tuple[str, ...]:
        attachment = self.bindings[request_id]["helper_attachment"]
        return () if not attachment else tuple(attachment.get("lease_tokens", ()))

    def _event(self, request_id, kind, at_us, detail):
        self.events.append((request_id, kind, at_us, dict(detail)))

    def _fraction(self, request_id, fraction, **_kwargs):
        binding = self.bindings[request_id]
        binding["fraction_ppm"] = fraction
        if binding["helper_attachment"]:
            binding["helper_attachment"]["lease_tokens"] = ()

    def _renew(self, request_id, *, lease_tokens, reserved_until_us, observed_at_us):
        self.bindings[request_id]["helper_attachment"]["lease_reserved_until_us"] = reserved_until_us

    def _reserve(self, resources, owner_id, at_us, finish_us):
        if self.live:
            raise UnifiedScheduleError("capacity-1 phone lanes are held")
        self.counter += 1
        rows = tuple(SimpleNamespace(token="lease-%d-%s" % (self.counter, name)) for name in resources)
        self.live.update(row.token for row in rows)
        self.events.append((owner_id, "RESERVED", at_us, {"tokens": [row.token for row in rows]}))
        return rows

    def _release(self, token, at_us):
        self.live.discard(token)


class Rig:
    """Real adaptive controller + real lease code + the fake server + (optionally) the real active hook."""

    def __init__(self, *, active: bool, start_s: float = 377.0) -> None:
        self.controller = AdaptiveDecodeController()
        self.config = adaptive_config()
        self.server = FakeLlamaServer(start_s)
        self.lanes = Lanes()
        self.slot_of: dict[str, int] = {}
        self.requests: dict[str, SimRequest] = {}
        self.first_token_s: dict[str, float] = {}
        self.end_s: dict[str, float] = {}
        self.decisions: list[tuple[str, str, int, bool]] = []
        self.active = JointPlannerActive(JointPlannerActiveConfig(budget_ms=500.0)) if active else None
        # the unified scheduler's registration, bound to this rig (the controller and the ticket checks)
        self.scheduler = SimpleNamespace(
            _adaptive_decode=self.controller, _adaptive_decode_config=self.config,
            _has_dormant_phone_ffn_runtime=lambda plan: False,
            _model_placement_controller=SimpleNamespace(
                _request_decode_progress={}, _phone_session_states={}),
        )
        for name in ("register_joint_join_prefill_yield", "clear_joint_join_prefill_yield",
                     "joint_join_prefill_yield_events"):
            setattr(self.scheduler, name, getattr(AdaptiveDecodeControlMixin, name).__get__(self.scheduler))

    # journal events

    def ticket(self, request_id: str, state: str):
        request = self.requests[request_id]
        return SimpleNamespace(
            request=SimpleNamespace(request_id=request_id, arrival_us=int(request.arrival_s * 1e6),
                                    input_tokens=request.input_tokens, output_tokens=request.output_tokens),
            model=SimpleNamespace(model_id=QWEN, artifact_sha256=ARTIFACT),
            decision=SimpleNamespace(start_us=int(self.server.now * 1e6), finish_upper_us=0),
            execution_plan=SimpleNamespace(
                route_id="auto:coordinated:physical:hot:desktop:residency:hot", transitions=(),
                desktop_placement_sha256=PLACEMENT, helper_envelope=None,
                execution_contract=SimpleNamespace(execution_mode="adaptive-split")),
            dispatch_state=state,
        )

    def journal(self, kind: str, request_id: str, state: str) -> None:
        if self.active is None:
            return
        progress = {rid: (slot["out"], slot["n"]) for slot in self.server.slots.values()
                    for rid in (slot["request_id"],)}
        phone = any(slot["phone"] and slot["state"] == "GENERATING" for slot in self.server.slots.values())
        self.scheduler._model_placement_controller._request_decode_progress = progress
        self.scheduler._model_placement_controller._phone_session_states = {
            "HTP" + str(i): SimpleNamespace(
                endpoint="session://op15-phone/HTP" + str(i), state="READY", resident_artifact_sha256=ARTIFACT,
                active_helper_references=("helper",) if phone else ())
            for i in range(3)}
        self.active.observe_ticket(self.scheduler, kind, self.ticket(request_id, state),
                                   int(round(self.server.now * 1e6)), state)

    # requests

    def arrive(self, request: SimRequest) -> None:
        self.requests[request.request_id] = request
        self.journal("DECISION", request.request_id, "QUEUED")

    def acquire(self, request_id: str, slot_id: int, *, decoded: int = 0, phone: bool = False) -> None:
        request = self.requests[request_id]
        self.slot_of[request_id] = slot_id
        self.server.admit(slot_id, request_id, request.input_tokens, request.output_tokens,
                          decoded=decoded, phone=phone)
        self.lanes.acquire(request_id)
        self.journal("ACQUIRED", request_id, "ACQUIRED")

    # adapter

    def handle(self, request_id: str, directive, token_index: int, at_us: int) -> None:
        while directive is not None and directive.control is not None:
            control = directive.control
            phone = not control.policy.baseline
            self.decisions.append((request_id, directive.reason, token_index, phone))
            self.server.control(self.slot_of[request_id], phone)
            self.lanes.apply(request_id, phone, token_index, at_us)
            directive = self.controller.acknowledge(request_id, AdaptiveDecodePolicyAck(
                request_id, self.slot_of[request_id], control.plan_generation, token_index, at_us,
                control.policy.policy_hash))

    def token(self, slot_id: int, request_id: str, n: int, done: bool) -> None:
        at_us = int(round(self.server.now * 1e6))
        if request_id not in self.first_token_s:
            self.first_token_s[request_id] = self.server.now
        sessions = self.controller._sessions
        if request_id not in sessions and request_id not in self.controller._completed and not done:
            directive = self.controller.start(
                request_id=request_id, ticket_id=request_id + ":0", model_artifact_sha256=ARTIFACT,
                planning_profile_sha256=PLAN, baseline=BASELINE, candidates=(FULL,),
                output_tokens=self.requests[request_id].output_tokens, context_length=64,
                active_batch=self.server.processing(), deadline_us=10**13, slot_id=slot_id,
                first_token_index=n, first_token_at_us=at_us, config=self.config,
                helper_layout_generation=1, helper_layout_geometry_sha256=GEOMETRY,
                helper_evidence_state="LEARNING")
            self.handle(request_id, directive, n, at_us)
        elif request_id in sessions:
            directive = self.controller.boundary(request_id, slot_id=slot_id, token_index=n, at_us=at_us,
                                                 terminal=done)
            if directive is not None and directive.boundary is not None:
                boundary = directive.boundary
                session = sessions[request_id]
                live = self.server.processing() + (1 if done else 0)
                directive = self.controller.record_window(request_id, boundary, observation(
                    40 if not boundary.policy.baseline else 100, boundary.token_count,
                    not boundary.policy.baseline, next_active_batch=live,
                    membership_changed=live != session.active_batch))
                if not done:
                    self.handle(request_id, directive, n, at_us)
        if done:
            self.end_s[request_id] = self.server.now
            self.controller.recover_for_restart(request_id, "FAKE_SERVER_DONE")
            self.lanes.complete(request_id, at_us)
            self.journal("COMPLETED", request_id, "COMPLETED")

    def run_until(self, t_s: float) -> None:
        while self.server.now < t_s:
            produced = self.server.step()
            if not produced:
                self.server.now = t_s
                return
            forward = self.server.forwards[-1]
            if forward["phone"]:  # the helper-window lease tokens each phone row holds in this forward
                forward["lease_tokens"] = {rid: self.lanes.tokens(rid) for rid in forward["decode"]}
            for slot_id, request_id, n, done in produced:
                self.token(slot_id, request_id, n, done)

    def run_to_end(self) -> None:
        self.run_until(math.inf)


S2A = {  # the s2a request shapes and times (RESULT.json / eval_v2_runs.json)
    "001": SimRequest("001", "qwen", 118.0, 160, 124),
    "003": SimRequest("003", "qwen", 404.0, 246, 262),
    "004": SimRequest("004", "qwen", 458.0, 106, 319),
}


def run_s2a_sequence(active: bool) -> Rig:
    """001 decodes from its first token at 381.2 s (s2a); 003 arrives 404.0 / is acquired 405.6;
    004 arrives 458.0 / is acquired 458.7 (the recorded times)."""
    rig = Rig(active=active, start_s=377.0)
    rig.arrive(S2A["001"])
    rig.acquire("001", 0)
    rig.run_until(404.0)
    rig.arrive(S2A["003"])
    rig.run_until(405.6)
    rig.acquire("003", 1)
    rig.run_until(458.0)
    rig.arrive(S2A["004"])
    rig.run_until(458.7)
    rig.acquire("004", 2)
    rig.run_to_end()
    return rig


def co_decoded(rig: Rig, a: str, b: str) -> list[dict]:
    return [f for f in rig.server.forwards if a in f["decode"] and b in f["decode"] and f["phone"]]


class S2aCoDecodeTests(unittest.TestCase):
    """(a) the s2a sequence: parked behind the holder today, co-decoded under the active mode."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sequential = run_s2a_sequence(active=False)
        cls.active = run_s2a_sequence(active=True)

    def test_the_model_reproduces_the_s2a_serialization_without_the_planner(self) -> None:
        rig = self.sequential
        self.assertGreaterEqual(rig.first_token_s["003"], rig.end_s["001"] - 1e-9)
        self.assertGreaterEqual(rig.first_token_s["004"], rig.end_s["003"] - 1e-9)
        self.assertFalse(co_decoded(rig, "001", "003"))
        # the holder reached its phone policy before the joiner arrived (s2a 001 ran 1,000,000 ppm)
        self.assertTrue(any(f["phone"] and f["decode"] == ["001"] for f in rig.server.forwards
                            if f["t"] < 404.0))
        # s2a: 003 first token 445.8 s, 004 576.1 s (the model: within a few seconds)
        self.assertAlmostEqual(rig.first_token_s["003"], 445.8, delta=8.0)
        self.assertAlmostEqual(rig.first_token_s["004"], 576.1, delta=15.0)

    def test_the_planner_joins_003_and_004_at_their_acquisition(self) -> None:
        records = [r for r in self.active.active.records() if "error" not in r]
        joined = {(r["request_id"], r["event_kind"], r["view"]): r for r in records if r["executed"]}
        self.assertEqual(sorted(joined), [("003", "ACQUIRED", "pre_acquisition"),
                                          ("004", "ACQUIRED", "pre_acquisition")])
        for record in joined.values():
            rid = record["request_id"]
            self.assertEqual(record["sequential"], ["park " + rid])
            self.assertEqual(record["planner"], ["admit " + rid])
            self.assertEqual(record["executed"], ["join " + rid + " (prefill yield)"])
            self.assertIsNone(record["fallback"])
            self.assertGreater(record["predicted"]["gain_j"], 50.0)
            self.assertLess(record["predicted"]["planner_score_j"], record["predicted"]["sequential_score_j"])
        decided = [r for r in records if r["event_kind"] == "DECISION" and r["request_id"] in ("003", "004")]
        self.assertEqual([r["deferred_to_acquisition"] for r in decided], [["003"], ["004"]])
        summary = self.active.active.summary()
        self.assertEqual(summary["executed_joins"], ["003", "004"])
        self.assertEqual(summary["errors"], 0)
        self.assertEqual(summary["prefill_yields"]["registered"], 2)
        self.assertEqual(summary["prefill_yields"]["ended_by_reason"], {"JOINER_STARTED": 2})

    def test_003_co_decodes_with_the_holder_under_the_shared_phone_policy(self) -> None:
        rig = self.active
        self.assertLess(rig.first_token_s["003"], rig.end_s["001"])
        # yield within one step + the prompt (246 tokens: 5.2 s) + one host step
        self.assertLess(rig.first_token_s["003"], 405.6 + 1.0 + 5.2 + 1.5)
        joint = co_decoded(rig, "001", "003")
        self.assertGreater(len(joint), 20)
        self.assertTrue(all(f["phone_rows"] == 2 for f in joint))  # one coalesced call, both rows
        # the holder yielded to the host exactly while 003 prefilled, then both returned to the phone
        holder = [(reason, phone) for rid, reason, _, phone in rig.decisions if rid == "001"]
        self.assertIn((PREFILL_YIELD_REASON, False), holder)
        prompt = next(f for f in rig.server.forwards if "003" in f["prompt"])
        self.assertFalse(prompt["phone"])
        self.assertIn("001", prompt["decode"])  # the prompt batched with the holder's host step

    def test_004_joins_the_running_phone_batch_as_well(self) -> None:
        rig = self.active
        self.assertLess(rig.first_token_s["004"], rig.end_s["003"])
        self.assertLess(rig.first_token_s["004"], 458.7 + 1.0 + 4.3 + 1.5)
        joint = co_decoded(rig, "003", "004")
        self.assertGreater(len(joint), 20)
        self.assertTrue(all(f["phone_rows"] == 2 for f in joint))

    def test_the_rows_share_the_holders_capacity_one_phone_lanes(self) -> None:
        rig = self.active
        # every phone forward that carries two rows runs on ONE helper-window reservation
        for a, b in (("001", "003"), ("003", "004")):
            joint = co_decoded(rig, a, b)
            tokens = {tuple(sorted({f["lease_tokens"][a], f["lease_tokens"][b]})) for f in joint}
            with self.subTest(pair=(a, b)):
                self.assertTrue(all(len(pair) == 1 and pair[0] for pair in tokens), tokens)
        shared = {(row[0], tuple(row[3]["member_request_ids"])) for row in rig.lanes.events
                  if row[1] == "SERVER_HELPER_LEASES_SHARED"}
        self.assertIn(("001", ("003",)), shared)  # the holder returns onto the joiner's reservation
        self.assertIn(("003", ("004",)), shared)
        # the capacity-1 lanes never have two holders (the fake refuses a second reservation)
        self.assertTrue([row for row in rig.lanes.events if row[1] == "RESERVED"])
        # the sequential run never shares: the joiner only takes the lanes after the holder ended
        self.assertFalse([row for row in self.sequential.lanes.events if row[1] == "SERVER_HELPER_LEASES_SHARED"])

    def test_the_active_run_finishes_the_three_requests_earlier(self) -> None:
        sequential, active = self.sequential, self.active
        self.assertLess(active.end_s["003"], sequential.end_s["003"] - 20.0)
        self.assertLess(active.end_s["004"], sequential.end_s["004"] - 60.0)
        # the holder pays the joiner's prompt forward (~5.8 s), one alternating round and b2 steps
        # (514 vs 489 ms), within the planner's 30 s per-epoch delay rule
        self.assertLessEqual(active.end_s["001"], sequential.end_s["001"] + 12.0)
        latency = {rid: active.end_s[rid] - S2A[rid].arrival_s for rid in S2A}
        base = {rid: sequential.end_s[rid] - S2A[rid].arrival_s for rid in S2A}
        self.assertLess(sum(latency.values()), sum(base.values()))


# Prefill yield in the controller ---------------------------------------------------------------------


class PrefillYieldControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = adaptive_config()

    def start(self, controller, request_id, slot_id, output_tokens, at_us, active_batch=1):
        return controller.start(
            request_id=request_id, ticket_id=request_id + ":0", model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN, baseline=BASELINE, candidates=(FULL,), output_tokens=output_tokens,
            context_length=64, active_batch=active_batch, deadline_us=10**12, slot_id=slot_id,
            first_token_index=1, first_token_at_us=at_us, config=self.config,
            helper_layout_generation=1, helper_layout_geometry_sha256=GEOMETRY, helper_evidence_state="LEARNING")

    def window(self, controller, request_id, at_us, *, energy=None, batch=None, terminal=False):
        session = controller._sessions[request_id]
        directive = controller.boundary(request_id, slot_id=session.slot_id, token_index=session.target_token,
                                        at_us=at_us, terminal=terminal)
        boundary = directive.boundary
        phone = not boundary.policy.baseline
        return controller.record_window(request_id, boundary, observation(
            energy if energy is not None else (40 if phone else 100), boundary.token_count, phone,
            next_active_batch=batch, membership_changed=batch is not None and batch != session.active_batch))

    def ack(self, controller, request_id, directive, at_us):
        session = controller._sessions[request_id]
        return controller.acknowledge(request_id, AdaptiveDecodePolicyAck(
            request_id, session.slot_id, directive.control.plan_generation, session.transition_start_token,
            at_us, directive.control.policy.policy_hash))

    def phone_holder(self, controller):
        """request-a qualifies the phone alone (batch 1), as 001 did in s2a."""
        self.start(controller, "request-a", 1, 200, 1_000)
        probe = self.window(controller, "request-a", 3_000)
        self.assertFalse(probe.control.policy.baseline)
        self.ack(controller, "request-a", probe, 3_000)
        self.assertIsNone(self.window(controller, "request-a", 5_000).control)
        self.assertFalse(controller.active_policy("request-a").baseline)

    def register(self, controller, joiner="request-b", at_us=5_500, expires_at_us=10**9):
        return controller.register_prefill_yield(
            joiner, model_artifact_sha256=ARTIFACT, desktop_placement_sha256=PLACEMENT,
            at_us=at_us, expires_at_us=expires_at_us)

    def test_registration_outcomes(self) -> None:
        controller = AdaptiveDecodeController()
        self.assertEqual(self.register(controller), ("NO_CO_TENANT_SESSION", ()))
        self.phone_holder(controller)
        self.assertEqual(controller.register_prefill_yield(
            "request-b", model_artifact_sha256="sha256:" + "9" * 64, desktop_placement_sha256=PLACEMENT,
            at_us=5_500, expires_at_us=10**9), ("NO_CO_TENANT_SESSION", ()))
        self.assertEqual(self.register(controller), ("REGISTERED", ("request-a",)))
        self.assertEqual(self.register(controller, joiner="request-a"), ("JOINER_ALREADY_DECODING", ()))
        for bad in (dict(at_us=5, expires_at_us=5), dict(at_us=-1, expires_at_us=5)):
            with self.subTest(bad=bad), self.assertRaises(Exception):
                self.register(controller, **bad)

    def test_the_phone_window_closes_at_the_next_token_and_the_holder_runs_the_host(self) -> None:
        controller = AdaptiveDecodeController()
        self.phone_holder(controller)
        session = controller._sessions["request-a"]
        target = session.target_token
        # mid-window tokens close nothing without a registration
        self.assertIsNone(controller.boundary("request-a", slot_id=1, token_index=target - 2, at_us=5_600))
        self.register(controller, at_us=5_650)
        directive = controller.boundary("request-a", slot_id=1, token_index=target - 1, at_us=5_700)
        self.assertIsNotNone(directive.boundary)
        decided = controller.record_window("request-a", directive.boundary, observation(
            40, directive.boundary.token_count, True, next_active_batch=2, membership_changed=True))
        self.assertEqual(decided.reason, PREFILL_YIELD_REASON)
        self.assertTrue(decided.control.policy.baseline)
        opened = self.ack(controller, "request-a", decided, 5_800)
        self.assertIsNone(opened.control)
        # still yielding at the next window; the group keeps its phone verdict and proposal
        again = self.window(controller, "request-a", 7_000)
        self.assertEqual(again.reason, PREFILL_YIELD_REASON)
        self.assertIsNone(again.control)
        group = controller._server_policies[controller.shared_server_policy_key("request-a")]
        self.assertFalse(group.policy.baseline)
        self.assertEqual(dict(group.verdicts)[1], FULL)

    def test_the_joiner_follows_the_group_and_the_holder_returns_after_one_token(self) -> None:
        controller = AdaptiveDecodeController()
        self.phone_holder(controller)
        self.register(controller)
        yielded = self.window(controller, "request-a", 6_000, batch=2)
        self.ack(controller, "request-a", yielded, 6_100)
        joiner = self.start(controller, "request-b", 2, 100, 7_000, active_batch=2)
        self.assertFalse(joiner.control.policy.baseline)  # the group's phone policy at the first token
        events = [row["kind"] for row in controller.prefill_yield_events()]
        self.assertEqual(events, ["REGISTERED", "APPLIED", "ENDED"])
        self.assertEqual(controller.prefill_yield_events()[-1]["reason"], "JOINER_STARTED")
        # the holder's host window closes at its next token and returns to the group's phone policy
        session = controller._sessions["request-a"]
        directive = controller.boundary("request-a", slot_id=1, token_index=session.window_start_token + 1,
                                        at_us=7_100)
        self.assertIsNotNone(directive.boundary)
        back = controller.record_window("request-a", directive.boundary, observation(
            100, directive.boundary.token_count, False))
        self.assertFalse(back.control.policy.baseline)
        self.assertNotEqual(back.reason, PREFILL_YIELD_REASON)
        self.assertFalse(controller._prefill_yields)

    def test_a_request_local_phone_switch_is_held_on_the_host(self) -> None:
        controller = AdaptiveDecodeController()
        # alone and without a verdict: the first window's probe is a request-local phone control
        self.start(controller, "request-a", 1, 200, 1_000)
        self.register(controller, at_us=2_000)
        probe = self.window(controller, "request-a", 3_000)
        self.assertTrue(probe.control is None or probe.control.policy.baseline)
        self.assertEqual(controller.snapshot("request-a")["zero_assistance_reason"], PREFILL_YIELD_REASON)
        # the same controller without the registration probes the phone here
        plain = AdaptiveDecodeController()
        self.start(plain, "request-a", 1, 200, 1_000)
        self.assertFalse(self.window(plain, "request-a", 3_000).control.policy.baseline)

    def test_expiry_and_clear_end_the_yield(self) -> None:
        controller = AdaptiveDecodeController()
        self.phone_holder(controller)
        self.register(controller, at_us=5_500, expires_at_us=5_900)
        session = controller._sessions["request-a"]
        self.assertIsNone(controller.boundary("request-a", slot_id=1, token_index=session.target_token - 1,
                                              at_us=6_000))
        self.assertEqual(controller.prefill_yield_events()[-1]["reason"], "EXPIRED")
        self.register(controller, joiner="request-c", at_us=6_100)
        self.assertTrue(controller.clear_prefill_yield("request-c", at_us=6_200, reason="JOINER_CANCELLED"))
        self.assertFalse(controller.clear_prefill_yield("request-c", at_us=6_300, reason="JOINER_CANCELLED"))
        self.assertEqual(controller.prefill_yield_events()[-1]["reason"], "JOINER_CANCELLED")

    def test_a_yield_issues_host_policies_only_and_later_controls_match_the_unchanged_controller(self) -> None:
        """(d) the yield never issues a phone control; once it ended, the holder's directive is the
        one a controller that never yielded issues from the same state."""
        yielded, plain = AdaptiveDecodeController(), AdaptiveDecodeController()
        for controller in (yielded, plain):
            self.phone_holder(controller)
        self.register(yielded)
        first = self.window(yielded, "request-a", 6_000, batch=2)
        self.assertTrue(first.control.policy.baseline)
        self.ack(yielded, "request-a", first, 6_100)
        for controller in (yielded, plain):
            self.start(controller, "request-b", 2, 100, 7_000, active_batch=2)
        reference = self.window(plain, "request-a", 8_000, batch=2)
        session = yielded._sessions["request-a"]
        directive = yielded.boundary("request-a", slot_id=1, token_index=session.window_start_token + 1,
                                     at_us=8_000)
        resumed = yielded.record_window("request-a", directive.boundary, observation(
            100, directive.boundary.token_count, False))
        self.assertEqual(resumed.control.policy, FULL)
        self.assertEqual(plain.active_policy("request-a"), FULL)
        self.assertIsNone(reference.control)  # the unchanged controller already ran the same policy

    def test_without_a_registration_nothing_yields(self) -> None:
        """(c) the absent key: the registry stays empty, the checks return at their first line and
        no directive carries the yield reason (the unchanged coherence suites cover the rest)."""
        from research_dev.scheduler._internal.adaptive_decode_ops import prefill_yield
        controller = AdaptiveDecodeController()
        self.phone_holder(controller)
        follower = self.start(controller, "request-b", 2, 100, 5_500, active_batch=2)
        self.ack(controller, "request-b", follower, 5_600)
        trail = [self.window(controller, "request-a", 6_000, batch=2), self.window(controller, "request-b", 6_500)]
        self.assertTrue(all(d.reason != PREFILL_YIELD_REASON for d in (follower, *trail)))
        for request_id in ("request-a", "request-b"):
            session = controller._sessions[request_id]
            self.assertIsNone(prefill_yield.pending(controller, session, 7_000))
            self.assertFalse(prefill_yield.closes_window(controller, session, 7_000))
        self.assertEqual((controller._prefill_yields, controller.prefill_yield_events()), ({}, ()))


# Classification and the hook's fallbacks --------------------------------------------------------------


def epoch(**overrides) -> ShadowEpoch:
    values = dict(
        sequence=1, event_kind="ACQUIRED", lifecycle_state="ACQUIRED", now_s=405.6, request_id="003",
        live={"planned_start_s": 404.0}, queue=(SimRequest("003", "qwen", 404.0, 246, 262),),
        rows=((SimRequest("001", "qwen", 118.0, 160, 124), 75.0),), resident="qwen", loading=None,
        load_end_s=0.0, assisted=True, phone_model="qwen", phone_loading=None, phone_ready_s=0.0,
    )
    values.update(overrides)
    return ShadowEpoch(**values)


def plan(actions, incumbent, **overrides) -> PlanResult:
    values = dict(actions=tuple(actions), incumbent=tuple(incumbent), chosen_index=0 if actions == incumbent else 1,
                  scores=(100.0, 50.0), candidates=(tuple(incumbent), tuple(actions)), feasible=(True, True),
                  elapsed_ms=1.0, budget_exhausted=False, fallback_reason=None, rollouts=2,
                  predicted_gain_j=50.0, energies=(1000.0, 900.0))
    values.update(overrides)
    return PlanResult(**values)


class ClassificationTests(unittest.TestCase):
    def test_join_into_an_assisted_batch_and_the_advisory_rest(self) -> None:
        sim = simulator_from_epoch(epoch(), COST)
        decided = classify_plan(plan([Admit(("003",))], [Park(("003",))]), sim)
        self.assertEqual(decided, {"assisted_batch": True, "joins": ["003"], "advisory": []})
        mixed = classify_plan(plan([Admit(("003",)), Provision("op15", "gemma")], [Park(("003",))]), sim)
        self.assertEqual(mixed["joins"], ["003"])
        self.assertEqual(mixed["advisory"], ["provision op15 gemma"])
        switch = classify_plan(plan([Switch("gemma")], []), sim)
        self.assertEqual((switch["joins"], switch["advisory"]), ([], ["switch gemma"]))

    def test_a_host_only_batch_needs_no_join(self) -> None:
        sim = simulator_from_epoch(epoch(assisted=False, phone_model=None), COST)
        decided = classify_plan(plan([Admit(("003",))], [Park(("003",))]), sim)
        self.assertEqual(decided["joins"], [])
        self.assertEqual(decided["advisory"], ["admit:003", "park:003"])

    def test_parked_rows_wait_in_the_simulator(self) -> None:
        sim = simulator_from_epoch(epoch(rows=((SimRequest("001", "qwen", 118.0, 160, 124), 75.0),
                                               (SimRequest("002", "qwen", 400.0, 50, 90), 90.0)),
                                         parked=("002",)), COST)
        self.assertEqual([row.parked for row in sim.server.rows], [False, True])
        alone = simulator_from_epoch(epoch(parked=("001",)), COST)
        self.assertEqual([row.parked for row in alone.server.rows], [False])


class HookStub:
    """A scheduler stand-in for the hook: lock-free placement fields and the registration methods."""

    def __init__(self, outcome="REGISTERED", assisted=True, progress=None):
        self.calls = []
        self.outcome = outcome
        self._model_placement_controller = SimpleNamespace(
            _request_decode_progress=progress if progress is not None else {"001": (124, 50)},
            _phone_session_states={"HTP0": SimpleNamespace(
                endpoint="session://op15-phone/HTP0", state="READY", resident_artifact_sha256="sha256:" + QWEN,
                active_helper_references=("h",) if assisted else ())})

    def register_joint_join_prefill_yield(self, ticket, *, at_us, expires_at_us):
        self.calls.append(("register", ticket.request.request_id, at_us, expires_at_us))
        return {"outcome": self.outcome, "co_tenant_request_ids": ["001"]}

    def clear_joint_join_prefill_yield(self, request_id, *, at_us, reason):
        self.calls.append(("clear", request_id, reason))
        return True

    def joint_join_prefill_yield_events(self):
        return ({"kind": "REGISTERED", "joiner_request_id": "003", "observed_at_us": 1},)


def stub_ticket(request_id, arrival_s, output_tokens, *, start_s=None, model_id=QWEN, input_tokens=100):
    return SimpleNamespace(
        request=SimpleNamespace(request_id=request_id, arrival_us=int(arrival_s * 1e6), input_tokens=input_tokens,
                                output_tokens=output_tokens),
        model=SimpleNamespace(model_id=model_id, artifact_sha256="sha256:" + model_id),
        decision=SimpleNamespace(start_us=int((start_s if start_s is not None else arrival_s) * 1e6),
                                 finish_upper_us=0),
        execution_plan=SimpleNamespace(route_id="route", transitions=()),
        dispatch_state="QUEUED",
    )


class HookTests(unittest.TestCase):
    def feed(self, active, stub, *, joiner_start_s=404.0, acquire=True):
        active.observe_ticket(stub, "DECISION", stub_ticket("001", 118.0, 124), 290_000_000, "QUEUED")
        active.observe_ticket(stub, "ACQUIRED", stub_ticket("001", 118.0, 124), 377_000_000, "ACQUIRED")
        active.observe_ticket(stub, "DECISION", stub_ticket("003", 404.0, 262, start_s=joiner_start_s,
                                                            input_tokens=246), 404_000_000, "QUEUED")
        if acquire:
            active.observe_ticket(stub, "ACQUIRED", stub_ticket("003", 404.0, 262, input_tokens=246),
                                  405_600_000, "ACQUIRED")
        return [r for r in active.records() if r.get("request_id") == "003" or "error" in r]

    def test_the_join_is_executed_at_the_acquisition_with_a_bounded_yield(self) -> None:
        active, stub = JointPlannerActive(JointPlannerActiveConfig(budget_ms=500.0)), HookStub()
        decision, acquired = self.feed(active, stub)
        self.assertEqual(decision["deferred_to_acquisition"], ["003"])
        self.assertEqual(acquired["executed"], ["join 003 (prefill yield)"])
        self.assertEqual(acquired["execution_detail"]["prefill_yield"]["co_tenant_request_ids"], ["001"])
        (_, joiner, at_us, expires_at_us), = stub.calls
        self.assertEqual((joiner, at_us), ("003", 405_600_000))
        prefill = COST.model("qwen").prefill_s(246)
        self.assertAlmostEqual((expires_at_us - at_us) / 1e6, 10.0 + 3.0 * prefill, places=3)
        active.observe_ticket(stub, "COMPLETED", stub_ticket("003", 404.0, 262), 500_000_000, "COMPLETED")
        self.assertEqual(stub.calls[-1], ("clear", "003", "JOINER_COMPLETED"))
        summary = active.summary()
        self.assertEqual(summary["schema"], JOINT_PLANNER_ACTIVE_SCHEMA)
        self.assertEqual(summary["executed_joins"], ["003"])
        self.assertEqual(summary["prefill_yields"]["registered"], 1)
        artifact = active.artifact()
        json.dumps(artifact)
        self.assertEqual(len(artifact["prefill_yield_events"]), 1)
        result = active.result()
        self.assertEqual(len(result["epoch_decisions"]), summary["epochs"] + summary["errors"])
        row = next(r for r in result["epoch_decisions"] if r["executed"])
        self.assertEqual((row["request_id"], row["sequential"], row["planner"], row["fallback"]),
                         ("003", ["park 003"], ["admit 003"], None))
        self.assertLess(row["planner_score_j"], row["sequential_score_j"])
        json.dumps(result)

    def test_refused_registrations_fall_back_with_their_reason(self) -> None:
        for outcome in ("SERVER_POLICY_COHERENCE_DISABLED", "JOINER_NOT_ADAPTIVE", "NO_CO_TENANT_SESSION"):
            with self.subTest(outcome=outcome):
                active = JointPlannerActive(JointPlannerActiveConfig(budget_ms=500.0))
                acquired = self.feed(active, HookStub(outcome=outcome))[-1]
                self.assertEqual(acquired["executed"], [])
                self.assertEqual(acquired["fallback"]["kind"], FALLBACK)
                self.assertEqual(acquired["fallback"]["reason"], outcome)
                self.assertEqual(active.summary()["fallbacks_by_reason"], {outcome: 1})

    def test_a_scheduler_without_the_registration_falls_back(self) -> None:
        stub = HookStub()
        stub.register_joint_join_prefill_yield = None
        active = JointPlannerActive(JointPlannerActiveConfig(budget_ms=500.0))
        acquired = self.feed(active, SimpleNamespace(_model_placement_controller=stub._model_placement_controller))[-1]
        self.assertEqual(acquired["fallback"]["reason"], "SCHEDULER_WITHOUT_PREFILL_YIELD")

    def test_budget_expiry_falls_back_to_the_sequential_decision(self) -> None:
        # a real plan cut at the budget keeps the incumbent; nothing is executed
        active, stub = JointPlannerActive(JointPlannerActiveConfig(budget_ms=0.001)), HookStub()
        acquired = self.feed(active, stub)[-1]
        self.assertEqual(stub.calls, [])
        self.assertTrue(acquired["budget_expired"])
        self.assertEqual(acquired["fallback"]["reason"], "BUDGET_EXPIRED")
        self.assertEqual(acquired["executed"], [])
        # a join decided after the budget expired is not executed either
        active, stub = JointPlannerActive(JointPlannerActiveConfig(budget_ms=50.0)), HookStub()
        active.planner.plan = lambda sim: plan([Admit(("003",))], [Park(("003",))], elapsed_ms=75.0)
        acquired = self.feed(active, stub)[-1]
        self.assertEqual(acquired["joins"], ["003"])
        self.assertEqual((acquired["executed"], acquired["fallback"]["reason"]), ([], "BUDGET_EXPIRED"))
        self.assertEqual(stub.calls, [])
        self.assertEqual(active.summary()["budget_expired"], 4)

    def test_a_planner_fallback_is_recorded(self) -> None:
        active, stub = JointPlannerActive(JointPlannerActiveConfig(budget_ms=500.0)), HookStub()
        active.planner.plan = lambda sim: plan([Park(("003",))], [Park(("003",))],
                                               fallback_reason="SEARCH_FAILED: x", scores=(None,))
        acquired = self.feed(active, stub)[-1]
        self.assertEqual(acquired["fallback"], {"kind": FALLBACK, "reason": "PLANNER_FAILED",
                                                "detail": "SEARCH_FAILED: x"})
        self.assertEqual(stub.calls, [])

    def test_planner_failure_and_capture_failure_fall_back(self) -> None:
        active, stub = JointPlannerActive(JointPlannerActiveConfig(budget_ms=500.0)), HookStub()
        active.planner.plan = lambda sim: (_ for _ in ()).throw(RuntimeError("boom"))
        active.observe_ticket(stub, "DECISION", stub_ticket("001", 118.0, 124), 1, "QUEUED")
        active.observe_ticket(stub, "DECISION", SimpleNamespace(request=None), 2, "QUEUED")
        errors = [r for r in active.records() if "error" in r]
        self.assertEqual([r["error"] for r in errors], ["PLAN_FAILED", "CAPTURE_FAILED"])
        self.assertTrue(all(r["fallback"]["kind"] == FALLBACK for r in errors))
        self.assertEqual(active.summary()["errors"], 2)
        self.assertEqual(stub.calls, [])

    def test_a_decision_the_cascade_does_not_admit_is_not_overridden(self) -> None:
        active, stub = JointPlannerActive(JointPlannerActiveConfig(budget_ms=500.0)), HookStub()
        decision = self.feed(active, stub, joiner_start_s=700.0, acquire=False)[-1]
        self.assertEqual(decision["planner"], ["admit 003"])
        self.assertEqual(decision["fallback"]["reason"], "CASCADE_DID_NOT_ADMIT")
        self.assertEqual(stub.calls, [])

    def test_the_holder_without_phone_needs_no_join(self) -> None:
        active, stub = JointPlannerActive(JointPlannerActiveConfig(budget_ms=500.0)), HookStub(assisted=False)
        acquired = self.feed(active, stub)[-1]
        self.assertEqual(acquired["joins"], [])
        self.assertEqual(stub.calls, [])

    def test_advisory_deviations_are_recorded_not_executed(self) -> None:
        active = JointPlannerActive(JointPlannerActiveConfig(budget_ms=500.0))
        fake = plan([Switch("gemma")], [], elapsed_ms=1.0)
        active.planner.plan = lambda sim: fake
        stub = HookStub()
        active.observe_ticket(stub, "DECISION", stub_ticket("002", 312.0, 614, model_id="gemma-4-12b-q40-dequant-f16"),
                              312_000_000, "QUEUED")
        record = active.records()[-1]
        self.assertEqual(record["fallback"]["reason"], "ADVISORY_ONLY")
        self.assertEqual(record["fallback"]["detail"]["advisory"], ["switch gemma"])
        self.assertEqual(record["executed"], [])
        self.assertEqual(stub.calls, [])


class UnifiedRegistrationTests(unittest.TestCase):
    """The scheduler-side registration refuses what server policy coherence cannot express."""

    def owner(self, coherence=True, dormant=False):
        controller = AdaptiveDecodeController()
        owner = SimpleNamespace(_adaptive_decode=controller,
                                _adaptive_decode_config=adaptive_config(server_policy_coherence=coherence,
                                                                        batch_growth_verdict_inheritance=False),
                                _has_dormant_phone_ffn_runtime=lambda plan: dormant)
        return owner, controller

    def ticket(self, state="ACQUIRED", mode="adaptive-split", placement=PLACEMENT, helper=None):
        return SimpleNamespace(
            request=SimpleNamespace(request_id="003"), model=SimpleNamespace(artifact_sha256=ARTIFACT),
            dispatch_state=state, execution_plan=SimpleNamespace(
                desktop_placement_sha256=placement, helper_envelope=helper,
                execution_contract=SimpleNamespace(execution_mode=mode)))

    def test_outcomes(self) -> None:
        register = AdaptiveDecodeControlMixin.register_joint_join_prefill_yield
        owner, _ = self.owner(coherence=False)
        self.assertEqual(register(owner, self.ticket(), at_us=1, expires_at_us=2)["outcome"],
                         "SERVER_POLICY_COHERENCE_DISABLED")
        owner, controller = self.owner()
        cases = (
            (self.ticket(state="QUEUED"), "JOINER_NOT_ACQUIRED"),
            (self.ticket(placement=None), "JOINER_WITHOUT_DESKTOP_PARENT"),
            (self.ticket(mode="static-split"), "JOINER_NOT_ADAPTIVE"),
            (self.ticket(), "NO_CO_TENANT_SESSION"),
        )
        for ticket, outcome in cases:
            with self.subTest(outcome=outcome):
                self.assertEqual(register(owner, ticket, at_us=1, expires_at_us=2)["outcome"], outcome)
        self.assertEqual(register(owner, self.ticket(), at_us=5, expires_at_us=5)["outcome"],
                         "REGISTRATION_REJECTED")
        self.assertEqual(AdaptiveDecodeControlMixin.joint_join_prefill_yield_events(owner), ())


# Configuration, runner, shadow / absent unchanged, no check skipped ----------------------------------


class ConfigurationTests(unittest.TestCase):
    def test_active_configuration(self) -> None:
        config = JointPlannerActiveConfig.from_json({"mode": "active"})
        self.assertEqual((config.budget_ms, config.max_prefill_yield_s), (50.0, 60.0))
        self.assertIsInstance(joint_planner_config_from_json({"mode": "active", "budget_ms": 20}),
                              JointPlannerActiveConfig)
        self.assertIsInstance(joint_planner_config_from_json({"mode": "shadow"}), JointPlannerShadowConfig)
        for value in ({"mode": "active", "max_prefill_yield_s": 0}, {"mode": "active", "extra": 1},
                      {"mode": "active", "depth": -1}, {"mode": "active", "objective": "latency"},
                      {"mode": "replay"}, {}, ["active"]):
            with self.subTest(value=value), self.assertRaises(JointPlannerShadowError):
                joint_planner_config_from_json(value)
        self.assertTrue(issubclass(JointPlannerActiveError, JointPlannerShadowError))
        rest, parsed = joint_planner_from_policy_json({"event_replanning": True, "joint_planner": {"mode": "active"}})
        self.assertEqual(rest, {"event_replanning": True})
        self.assertIsInstance(parsed, JointPlannerActiveConfig)

    def test_campaign_accepts_the_active_key(self) -> None:
        checked = _dispatch_policy({"event_replanning": True,
                                    "joint_planner": {"mode": "active", "max_prefill_yield_s": 30}})
        self.assertEqual(checked["joint_planner"], {"max_prefill_yield_s": 30, "mode": "active"})
        with self.assertRaises(SchedulerConfigurationError):
            _dispatch_policy({"joint_planner": {"mode": "active", "max_prefill_yield_s": -1}})

    def test_runner_builds_the_active_planner_only_for_mode_active(self) -> None:
        active = json.dumps({"event_replanning": True, "joint_planner": {"mode": "active"}})
        shadow = json.dumps({"joint_planner": {"mode": "shadow"}})
        self.assertIsInstance(runner._joint_planner_active_from_json(active), JointPlannerActive)
        self.assertIsNone(runner._joint_planner_shadow_from_json(active))
        self.assertIsNone(runner._joint_planner_active_from_json(shadow))
        self.assertIsNone(runner._joint_planner_active_from_json(None))
        self.assertTrue(runner._dispatch_policy_from_json(active).event_replanning)
        self.assertIsNone(runner._dispatch_policy_from_json(json.dumps({"joint_planner": {"mode": "active"}})))
        self.assertEqual(runner._joint_planner_active_result(SimpleNamespace(joint_planner_active=lambda: None)), {})

    def test_runner_refuses_the_active_mode_without_server_policy_coherence(self) -> None:
        for config in (None, adaptive_config(server_policy_coherence=False, batch_growth_verdict_inheritance=False)):
            with self.subTest(config=config), self.assertRaisesRegex(Exception, "server_policy_coherence"):
                runner._require_joint_planner_active_prerequisites(config)
        runner._require_joint_planner_active_prerequisites(adaptive_config())


class ShadowUnchangedTests(unittest.TestCase):
    """(c) shadow mode output is unchanged by the active code."""

    def test_the_s2a_shadow_replay_still_reports_the_same_deviations(self) -> None:
        rows, summary = evaluation.shadow_replay_rows(evaluation.load_runs()["s2a"], COST)
        self.assertEqual((summary["epochs"], summary["errors"], summary["deviations"]), (42, 0, 4))
        self.assertEqual([row[:5] for row in rows], [
            ["404.0", "DECISION", "003", "park 003", "admit 003"],
            ["458.0", "DECISION", "004", "park 004", "admit 004"],
            ["1496.0", "DECISION", "010", "park 010", "admit 010"],
            ["1497.0", "DECISION", "llama00", "park 010", "admit 010"],
        ])
        self.assertNotIn("parked", json.dumps(summary))

    def test_the_shadow_epoch_default_parks_nothing(self) -> None:
        sim = simulator_from_epoch(epoch(), COST)
        self.assertEqual([row.parked for row in sim.server.rows], [False])
        self.assertEqual(sim.server.rows[0].decode_start_s, 405.6)


class ActiveEvaluationTests(unittest.TestCase):
    """The offline evaluator's executed policy: on longtail_eval_v2 (cool OP15) the joins are the
    planner's whole benefit, and a join lag of a few seconds keeps it."""

    def test_trace_joins(self) -> None:
        from research_dev.scheduler._internal.joint_planner import JointPlanner, JointPlannerConfig
        from research_dev.scheduler._internal.joint_planner_sim import SequentialPolicy, SimOptions
        requests = evaluation.trace_requests(evaluation.load_runs()["s2a"])
        results = {}
        for label, arm in (("sequential", SequentialPolicy()),
                              ("active", active_evaluation.ActiveExecutedPolicy()),
                              ("lagged", active_evaluation.ActiveExecutedPolicy(lag_s=3.0)),
                              ("planner", JointPlanner(JointPlannerConfig(budget_ms=250.0)))):
            results[label] = (arm, evaluation.metrics(evaluation.run_policy(requests, arm, COST, SimOptions())))
        self.assertEqual([rid for _, rid in results["active"][0].joins], ["003", "004", "006"])
        self.assertAlmostEqual(results["active"][1]["host_kj"], results["planner"][1]["host_kj"], places=6)
        self.assertLess(results["active"][1]["host_kj"], results["sequential"][1]["host_kj"])
        self.assertLess(results["active"][1]["p50_s"], results["sequential"][1]["p50_s"] - 30.0)
        self.assertLess(results["lagged"][1]["host_kj"], results["sequential"][1]["host_kj"])


class LiveSchedulerTests(unittest.TestCase):
    """(d) the real scheduler: with the active mode on, every admission, lease and identity decision
    (the decision log bytes) is the one the cascade takes; the planner only records and registers."""

    def run_scheduler(self, active):
        case = dispatch_harness.DispatchPolicySchedulerTests("test_same_model_arrival_joins_the_running_server")
        case.setUp()
        self.addCleanup(case.directory.cleanup)
        scheduler, model_a, model_b, hot = case.scheduler(dispatch_harness.WORK_CONSERVING)
        if active is not None:
            scheduler.configure_joint_planner_active(active)
        tickets = [
            case.submit(scheduler, model_a, hot, "a1", 1_000, 640),
            case.submit(scheduler, model_b, hot, "b1", 1_100, 8),
            case.submit(scheduler, model_a, hot, "a2", 1_200, 4),
        ]
        return scheduler, tickets

    def test_decisions_are_byte_identical_with_the_active_mode(self) -> None:
        plain, plain_tickets = self.run_scheduler(None)
        keys = {"model-a": "qwen", "model-b": "gemma"}
        active = JointPlannerActive(JointPlannerActiveConfig(), model_key=keys.get)
        observed, observed_tickets = self.run_scheduler(active)
        self.assertIsNone(plain.joint_planner_active())
        self.assertIs(observed.joint_planner_active(), active)
        self.assertEqual(plain.runtime_decision_log_bytes(), observed.runtime_decision_log_bytes())
        self.assertEqual([t.decision.start_us for t in plain_tickets],
                         [t.decision.start_us for t in observed_tickets])
        records = active.records()
        self.assertGreaterEqual(len(records), 3)
        self.assertTrue(all("error" not in r for r in records), records)
        self.assertEqual(observed.joint_join_prefill_yield_events(), ())

    def test_the_planner_code_calls_no_admission_lease_or_residency_function(self) -> None:
        allowed = {"register_joint_join_prefill_yield", "clear_joint_join_prefill_yield",
                   "joint_join_prefill_yield_events"}
        forbidden = ("reserve", "release", "extend_lease", "attach", "admit_", "submit", "replan",
                     "reprovision", "commit", "acquire", "dispatch", "cancel", "thermal")
        for name in ("_internal/joint_planner_active.py", "_internal/adaptive_decode_ops/prefill_yield.py"):
            tree = ast.parse((ROOT / name).read_text(encoding="ascii"))
            called = {node.func.attr for node in ast.walk(tree)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
            scheduler_calls = {c for c in called if any(word in c for word in forbidden)} - allowed
            with self.subTest(module=name):
                self.assertEqual(scheduler_calls, set())


if __name__ == "__main__":
    unittest.main()
