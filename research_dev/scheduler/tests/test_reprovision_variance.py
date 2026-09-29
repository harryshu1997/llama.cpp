#!/usr/bin/env python3
"""Run-to-run variance of the two-phone re-provisioning arm (longtail_eval_v2).

Two runs of the same arm (OP15 re-provisioned between Gemma and Qwen, static
Pixel co-helper on Qwen layers 18-23) differed by 28 % host energy because
Qwen ran almost unassisted in run 2.  ``data/reprov_variance_r2.json`` holds
the recorded facts (extracted from both RESULT dumps) these tests replay:

* (b) a transient THERMAL_LIMIT on every phone route at the Qwen desktop
  launch made the dormant FFN runtime fall back to a lexicographic order that
  put the unqualified split-row executor first; server-policy coherence then
  filtered every later READY layout to split-row: "ready layout produced no
  helper opportunity" for the whole phase although the geometry matched.
* (c) the two-phone envelope always carries the co-helper's layers, so the
  additive check (``assisted_layer_mask == primary layout mask``) never held:
  a request attached before the third session loaded could not expand
  ("request helper envelope identity is immutable").
* (a) a same-geometry PROPOSAL_UPDATED dropped the stamped replacement
  source, so for the whole load every waiting request re-derived it from the
  DRAINING source layout: "phone helper replacement source is not ready".
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from research_dev.scheduler import UnifiedScheduler
from research_dev.scheduler._internal.model_placement_controller import (
    ModelPlacementController,
    ModelPlacementControllerError,
    ModelPlacementPolicy,
    RequestHelperEnvelopeBinding,
)
from research_dev.scheduler._internal.phone_shards import (
    PhoneFfnResidencyLayout,
    PhoneFfnShardPlacement,
    _residency_geometry_sha256,
)
from research_dev.scheduler._internal.types import canonical_sha256
from research_dev.scheduler._unified.automated_selection import (
    AutomatedSelectionMixin,
)
from research_dev.scheduler._unified.automated_selection_ops import (
    dormant as dormant_ops,
    helpers as helper_ops,
)
from research_dev.scheduler._unified.common import _StalePhoneSessionAssignment
from research_dev.scheduler.config import (
    PhoneResidentModelReprovisioningConfiguration,
)
from research_dev.scheduler._unified.helper_envelopes_ops import (
    materialization as materialization_ops,
    policies as policy_ops,
    templates as template_ops,
)


FIXTURE = json.loads(
    (Path(__file__).resolve().parent / "data" / "reprov_variance_r2.json")
    .read_text()
)
QWEN = FIXTURE["qwen_artifact_sha256"]
PLACEMENT = canonical_sha256({"desktop-placement": "42a30600"})
BASELINE_ROUTE = FIXTURE["r2_qwen_phase_b_launch"]["desktop_route_id"]
BASELINE_EXECUTOR = "physical:hot:desktop"
PIXEL_LAYERS = 0xFC0000  # Qwen layers 18-23 on the static Pixel co-helper


def identity(label: str) -> str:
    return canonical_sha256({"identity": label})


def batch_plan_of(route_id: str) -> str:
    return "coalesced-batch" if ":coalesced-batch:" in route_id else "split-row"


def primary_mask_of(route_id: str) -> int:
    sessions = 3 if ":sessions:3:" in route_id else 2 if ":sessions:2:" in route_id else 1
    return (1 << (6 * sessions)) - 1


def qualified_executor_ids() -> frozenset[str]:
    """catalog_materialization: only the qualified batch plan is QUALIFIED."""

    return frozenset(
        row["executor_id"]
        for row in FIXTURE["r2_qwen_phase_b_launch"]["candidates"]
        if row["executor_id"].endswith(":coalesced-batch")
    )


def dormant_parameters(route_id: str) -> dict[str, object]:
    plan = batch_plan_of(route_id)
    return {
        "ffn_activation": "f16",
        "ffn_assistance_phase": "decode",
        "ffn_max_tokens": 4,
        "ffn_n_embd": 5120,
        "ffn_resident_columns": 17408,
        "ffn_resident_layer_mask": primary_mask_of(route_id) | PIXEL_LAYERS,
        "ffn_runtime_control_protocol": "decode-boundary-v1",
        "ffn_timeout_ms": 30000,
        "ffn_transport": "functionfs-usb",
        "phone_device_id": "op15-phone",
        "usb_allocator": "dmabuf",
        "usb_batch_plan": plan,
        "usb_full_duplex": True,
        "usb_max_payload_bytes": 40960,
        "usb_product_id": 1,
        "usb_queue_depth": 4,
        "usb_slot_safety_bytes": 0,
        "usb_split_h2d": False,
        "usb_transport_generation": 2,
        "usb_transport_profile_id": "profile-" + plan,
        "usb_vendor_id": 1,
        "usbfs_available_bytes": 1 << 30,
    }


def launch_rows() -> tuple[SimpleNamespace, SimpleNamespace, tuple]:
    """The r2 Qwen phase-B cold launch (004 attempt 5) as recorded."""

    baseline = SimpleNamespace(
        candidate_id=BASELINE_ROUTE,
        baseline=True,
        assisted_operator_kind=None,
        route_family="whole",
        paired_baseline_route_id=None,
        binding=SimpleNamespace(executor_id=BASELINE_EXECUTOR, artifact_sha256=QWEN),
        plan=SimpleNamespace(
            desktop_placement_sha256=PLACEMENT,
            residency_variant="cold",
            execution_contract=SimpleNamespace(execution_mode="desktop"),
            adapter_parameters={},
        ),
        maturity="QUALIFIED",
        rejection_reasons=(),
    )
    rows = tuple(
        SimpleNamespace(
            candidate_id=row["route_id"],
            assisted_operator_kind="ffn",
            route_family="operator_split",
            paired_baseline_route_id=BASELINE_ROUTE,
            maturity=row["maturity"],
            rejection_reasons=tuple(row["rejection_reasons"]),
            binding=SimpleNamespace(
                executor_id=row["executor_id"], artifact_sha256=QWEN
            ),
            plan=SimpleNamespace(
                baseline_executor_id=BASELINE_EXECUTOR,
                desktop_placement_sha256=PLACEMENT,
                execution_contract=SimpleNamespace(
                    execution_mode="adaptive-split"
                ),
                adapter_parameters=dormant_parameters(row["route_id"]),
            ),
        )
        for row in FIXTURE["r2_qwen_phase_b_launch"]["candidates"]
    )
    return baseline, SimpleNamespace(candidates=(baseline, *rows)), rows


class RecordedFixtureTests(unittest.TestCase):
    def test_fixture_records_the_split_row_lock_in(self) -> None:
        plans = FIXTURE["dormant_batch_plan"]
        self.assertEqual(
            {key: row["usb_batch_plan"] for key, row in plans["r1"].items()},
            {"001": "coalesced-batch", "004": "coalesced-batch",
             "011": "coalesced-batch"},
        )
        self.assertEqual(
            {key: row["usb_batch_plan"] for key, row in plans["r2"].items()},
            {"001": "coalesced-batch", "004": "split-row", "011": "split-row"},
        )
        launch = FIXTURE["r2_qwen_phase_b_launch"]["candidates"]
        self.assertTrue(all("THERMAL_LIMIT" in row["rejection_reasons"] for row in launch))
        self.assertIn("sessions:3:geometry:7bf55ae9bb49", FIXTURE["r2_phase_c_no_opportunity"])
        self.assertNotIn("coalesced-batch", FIXTURE["r2_phase_c_no_opportunity"])


class DormantBatchPlanTests(unittest.TestCase):
    """(b) the desktop's dormant runtime must be one a qualified helper can use."""

    def test_recorded_fallback_order_picks_split_row_without_qualification(self) -> None:
        baseline, candidate_set, _rows = launch_rows()
        parameters = AutomatedSelectionMixin._dormant_phone_ffn_parent_parameters(
            candidate_set, baseline, SimpleNamespace(artifact_sha256=QWEN)
        )
        # The historical order (no qualification view) is unchanged: this is
        # exactly the r2 launch, split-row sorts before coalesced-batch.
        self.assertEqual(parameters[0]["usb_batch_plan"], "split-row")
        self.assertEqual(parameters[0]["ffn_resident_layer_mask"], 0xFFFFFF)

    def test_transient_thermal_limit_keeps_the_qualified_batch_plan(self) -> None:
        baseline, candidate_set, _rows = launch_rows()
        parameters = AutomatedSelectionMixin._dormant_phone_ffn_parent_parameters(
            candidate_set, baseline, SimpleNamespace(artifact_sha256=QWEN),
            qualified_executor_ids(),
        )
        self.assertEqual(parameters[0]["usb_batch_plan"], "coalesced-batch")
        self.assertEqual(parameters[0]["ffn_resident_layer_mask"], 0xFFFFFF)
        # The widest runtime still wins inside the qualified plan.
        self.assertEqual(
            [row["usb_batch_plan"] for row in parameters],
            ["coalesced-batch"] * 3 + ["split-row"] * 3,
        )

    def test_order_is_unchanged_when_every_or_no_executor_qualifies(self) -> None:
        baseline, candidate_set, rows = launch_rows()
        manifest = SimpleNamespace(artifact_sha256=QWEN)
        historical = AutomatedSelectionMixin._dormant_phone_ffn_parent_parameters(
            candidate_set, baseline, manifest
        )
        for qualified in (
            frozenset(row.binding.executor_id for row in rows),
            frozenset(),
        ):
            self.assertEqual(
                AutomatedSelectionMixin._dormant_phone_ffn_parent_parameters(
                    candidate_set, baseline, manifest, qualified
                ),
                historical,
            )

    def test_launch_under_thermal_limit_binds_a_coalesced_dormant_runtime(self) -> None:
        """Replay r2 004 attempt 5: no envelope, no READY or retained helper."""

        baseline, candidate_set, _rows = launch_rows()
        captured: list[str] = []
        catalog = SimpleNamespace(composite_executors=tuple(
            SimpleNamespace(
                executor_id=row["executor_id"],
                maturity=(
                    "QUALIFIED"
                    if row["executor_id"] in qualified_executor_ids()
                    else "SHADOW"
                ),
            )
            for row in FIXTURE["r2_qwen_phase_b_launch"]["candidates"]
        ))

        def bind(candidate_set, baseline, selected, encoded):
            captured.append(encoded)
            return candidate_set, baseline, selected

        controller = SimpleNamespace(
            _runtime_capabilities=catalog,
            _maximum_phone_sessions=3,
            _retained_request_helper_for_baseline=lambda *args: None,
            _authoritative_ready_helper_template=lambda **kwargs: None,
            _required_dormant_phone_ffn_keys=dormant_ops._required_dormant_phone_ffn_keys,
            _dormant_phone_ffn_parent_parameters=(
                AutomatedSelectionMixin._dormant_phone_ffn_parent_parameters
            ),
            _dormant_phone_ffn_storage_superset=lambda params, plan, manifest: dict(params),
            _baseline_with_dormant_phone_ffn=bind,
            _model_placement_controller=SimpleNamespace(
                planning_phone_layout=lambda: None
            ),
            _defer_unready_phone_selection=helper_ops._defer_unready_phone_selection,
        )
        controller._complete_dormant_phone_ffn_parameters = (
            lambda *sets: dormant_ops._complete_dormant_phone_ffn_parameters(
                controller, *sets
            )
        )
        with mock.patch.object(
            helper_ops, "adaptive_decode_policies",
            lambda *args, **kwargs: (None, (), None),
        ), mock.patch.object(
            helper_ops, "adaptive_candidate_set_for_parent",
            lambda candidate_set, baseline: candidate_set,
        ):
            helper_ops._desktop_with_async_phone_helper(
                controller, candidate_set, baseline, (), "ENERGY", SimpleNamespace(
                    request_id="burstgpt_longtail_eval_v2:004", output_tokens=319
                ),
                SimpleNamespace(artifact_sha256=QWEN), "energy-aware",
            )
        self.assertEqual(len(captured), 1)
        self.assertEqual(json.loads(captured[0])["usb_batch_plan"], "coalesced-batch")


@dataclass(frozen=True)
class _CandidateSet:
    candidates: tuple
    baseline_route_id: str
    recovery_fallback_route_id: str | None = None
    search_metadata: tuple = ()


def _shard(session_id: str, mask: int, geometry: str) -> SimpleNamespace:
    return SimpleNamespace(
        artifact_sha256=QWEN, session_id=session_id,
        endpoint="session://op15-phone/" + session_id, layer_mask=mask,
        maximum_columns=17408, resident_bytes=3 << 30,
        resident_geometry_sha256=geometry, operator_plan_sha256=identity(session_id),
    )


class ReadyLayoutOpportunityTests(unittest.TestCase):
    """(b) the READY layout, not the route's session count, decides the helper."""

    def setUp(self) -> None:
        self.three = tuple(
            _shard(session, mask, identity("q3-" + session))
            for session, mask in (("HTP0", 0x3F), ("HTP1", 0xFC0), ("HTP2", 0x3F000))
        )
        self.two = self.three[:2]
        self.layout3 = SimpleNamespace(layout=SimpleNamespace(
            shards=self.three, geometry_sha256=_residency_geometry_sha256(self.three)
        ))
        self.layout2 = SimpleNamespace(layout=SimpleNamespace(
            shards=self.two, geometry_sha256=_residency_geometry_sha256(self.two)
        ))

        def candidate(shards, geometry, plan):
            sessions = len(shards)
            return SimpleNamespace(
                candidate_id=(
                    "auto:coordinated:physical:hot:phone-assisted:operator_split:"
                    + ("" if plan == "split-row" else plan + ":")
                    + f"750000:sessions:{sessions}:geometry:{geometry[7:19]}"
                ),
                assisted_operator_kind="ffn", maturity="SHADOW",
                rejection_reasons=("COLD_RESIDENCY_BREAK_EVEN", "ROUTE_NOT_QUALIFIED"),
                binding=SimpleNamespace(executor_id=(
                    "physical:hot:phone-assisted:operator_split"
                    + ("" if plan == "split-row" else ":" + plan)
                )),
                plan=SimpleNamespace(
                    adapter_parameters={
                        "phone_shard_set_geometry_sha256": geometry,
                        "usb_batch_plan": plan,
                    },
                    execution_contract=SimpleNamespace(phone_shards=shards),
                ),
            )

        rows = tuple(
            candidate(shards, layout.layout.geometry_sha256, plan)
            for shards, layout in ((self.three, self.layout3), (self.two, self.layout2))
            for plan in ("split-row", "coalesced-batch")
        )
        self.generated = _CandidateSet(candidates=rows, baseline_route_id=BASELINE_ROUTE)

    def test_route_for_another_session_count_is_never_offered(self) -> None:
        kept = policy_ops._candidate_set_for_ready_phone_layout(
            self.generated, QWEN, self.layout2, usb_batch_plan="coalesced-batch"
        )
        self.assertEqual(
            [row.candidate_id.split(":geometry:")[0].rsplit(":", 1)[-1]
             for row in kept.candidates],
            ["2"],
        )
        self.assertEqual(
            kept.candidates[0].plan.adapter_parameters["usb_batch_plan"],
            "coalesced-batch",
        )

    def test_no_opportunity_names_the_dormant_batch_plan(self) -> None:
        """Replay r2 phase C: READY 7bf55 layout, split-row desktop runtime."""

        catalog = SimpleNamespace(composite_executor_by_id={
            "physical:hot:phone-assisted:operator_split": SimpleNamespace(
                maturity="SHADOW"),
            "physical:hot:phone-assisted:operator_split:coalesced-batch": (
                SimpleNamespace(maturity="QUALIFIED")),
        })
        controller = SimpleNamespace(
            _runtime_capabilities=catalog,
            _candidate_set_for_ready_phone_layout=(
                policy_ops._candidate_set_for_ready_phone_layout
            ),
        )
        note = materialization_ops._dormant_batch_plan_note(
            controller, self.generated, QWEN, self.layout3, "split-row"
        )
        self.assertIn("desktop dormant usb_batch_plan='split-row'", note)
        self.assertIn("helper executor qualified: False", note)
        self.assertIn("('coalesced-batch', True)", note)
        self.assertEqual(
            materialization_ops._dormant_batch_plan_note(
                controller, self.generated, QWEN, self.layout3, None
            ),
            "",
        )


def qwen_gemma_layout(
    assignments: tuple[str, str, str],
    current: tuple[str, str, str] | None = None,
) -> PhoneFfnResidencyLayout:
    """The OP15 sessions of the recorded arm: Qwen 0-5/6-11/12-17, Gemma 24 layers."""

    masks = {
        "q": (0x3F, 0xFC0, 0x3F000),
        "g": (0xFF, 0xFF00, 0xFF0000),
    }
    artifacts = {"q": QWEN, "g": identity("gemma-artifact")}
    shards = tuple(
        PhoneFfnShardPlacement(
            artifact_sha256=artifacts[label],
            session_id="HTP" + str(index),
            endpoint="session://op15-phone/HTP" + str(index),
            memory_resource_id="op15-ram:session:HTP" + str(index),
            operator_ids=("blk." + str(index) + ".ffn",),
            layer_mask=masks[label][index],
            maximum_columns=17408 if label == "q" else 15360,
            resident_bytes=3 << 30,
            resident_geometry_sha256=identity(label + "-geometry-" + str(index)),
            operator_plan_sha256=identity(label + "-plan-" + str(index)),
        )
        for index, label in enumerate(assignments)
    )
    changed = tuple(
        "HTP" + str(index)
        for index, label in enumerate(assignments)
        if current is None or current[index] != label
    )
    by_artifact: dict[str, int] = {}
    for shard in shards:
        by_artifact[shard.artifact_sha256] = by_artifact.get(shard.artifact_sha256, 0) + 1000
    return PhoneFfnResidencyLayout(
        shards=shards,
        queued_work_by_artifact={artifact: 100 for artifact in by_artifact},
        queue_benefit_by_artifact=by_artifact,
        queue_benefit_by_session={row.session_id: 1000 for row in shards},
        queue_benefit=3000,
        transition_cost=100 * len(changed),
        transition_cost_by_session={session_id: 100 for session_id in changed},
        objective=100 * len(changed) - 3000,
        objective_kind="queue_rough_compute_ops",
        changed_session_ids=changed,
        geometry_sha256=_residency_geometry_sha256(shards),
    )


class _ControllerCase(unittest.TestCase):
    def controller(self) -> ModelPlacementController:
        return ModelPlacementController(ModelPlacementPolicy(
            debounce_us=100_000,
            switch_hysteresis_us=100_000,
            phone_layout_confirmation_snapshots=2,
            phone_minimum_residency_us=100_000,
        ))

    def publish(self, controller, layout, at_us, label):
        state = controller.propose_phone_layout(
            layout,
            workspace_bytes=1024,
            shared_compute_resource_id="op15-htp",
            shared_transport_resource_ids=("op15-functionfs", "desktop-usb-root"),
            observed_at_us=at_us,
        )
        self.begin(controller, state.generation, at_us + 10, label)
        return controller.complete_phone_layout_transition(
            generation=state.generation,
            ticket_id=label,
            transition_ids=(label,),
            geometry_sha256=layout.geometry_sha256,
            projection_token_sha256=identity(label),
            finished_at_us=at_us + 1_000,
        )

    def begin(self, controller, generation, at_us, label):
        return controller.begin_phone_layout_transition(
            generation,
            ticket_id=label,
            transition_ids=(label,),
            ready_at_us=at_us + 990,
            projection_token_sha256=identity(label),
            workspace_bytes=1024,
            observed_at_us=at_us,
        )


class CoHelperExpansionTests(_ControllerCase):
    """(c) a request attached before the third session loads must expand."""

    RECORD = FIXTURE["r2_003_immutable"]

    def envelope(self, state, mask: int, sessions: tuple[str, ...]):
        recorded = self.RECORD["gen6_envelope_r1"]
        return RequestHelperEnvelopeBinding(
            route_id="qwen-helper-" + str(state.generation),
            operator_plan_sha256=identity("qwen-helper-plan-" + str(state.generation)),
            desktop_parent_route_id=BASELINE_ROUTE,
            desktop_placement_sha256=PLACEMENT,
            phone_layout_generation=state.generation,
            phone_layout_geometry_sha256=state.layout.geometry_sha256,
            activation_dtype="f16",
            assisted_layer_mask=mask,
            maximum_columns=recorded["maximum_columns"],
            allowed_fractions_ppm=tuple(recorded["allowed_fractions_ppm"]),
            phone_session_ids=sessions,
            resource_ids=tuple(recorded["resource_ids"]),
        )

    def attached(self, *, co_helper: int = PIXEL_LAYERS):
        controller = self.controller()
        qqg = qwen_gemma_layout(("q", "q", "g"))
        gen5 = self.publish(controller, qqg, 100, "load-qqg")
        before = self.envelope(gen5, 0xFFF | co_helper, ("HTP0", "HTP1"))
        if co_helper == PIXEL_LAYERS:
            self.assertEqual(before.assisted_layer_mask, self.RECORD["gen5_envelope"]["assisted_layer_mask"])
        controller.bind_dispatched_request(
            "burstgpt_longtail_eval_v2:003", QWEN, BASELINE_ROUTE,
            identity("qwen-desktop"), 0,
            desktop_placement_sha256=PLACEMENT, kv_cache_owner_id="003:attempt:1",
            sequence_identity="003:attempt:1", server_slot_id=1,
            helper_envelope=before,
        )
        controller.mark_request_acquired("burstgpt_longtail_eval_v2:003")
        attachment = self.RECORD["attachment"]
        controller.attach_request_helper(
            "burstgpt_longtail_eval_v2:003",
            phone_layout_generation=gen5.generation,
            phone_layout_geometry_sha256=qqg.geometry_sha256,
            resident_component_identity_sha256=gen5.resident_component_identity_sha256,
            operator_plan_sha256=before.operator_plan_sha256,
            start_token_index=164,
            fraction_ppm=attachment["fraction_ppm"],
            lease_tokens=("lease-129",),
            lease_reserved_until_us=300_000,
            observed_at_us=1_200,
        )
        controller.record_request_helper_work(
            "burstgpt_longtail_eval_v2:003", attachment["completed_phone_calls"],
            observed_at_us=1_300,
        )
        qqq = qwen_gemma_layout(("q", "q", "q"), current=("q", "q", "g"))
        gen6 = self.publish(controller, qqq, 200_000, "load-htp2")
        return controller, gen6

    def test_recorded_two_phone_expansion_is_additive(self) -> None:
        controller, gen6 = self.attached()
        after = self.envelope(gen6, 0x3FFFF | PIXEL_LAYERS, ("HTP0", "HTP1", "HTP2"))
        self.assertEqual(after.assisted_layer_mask, self.RECORD["gen6_envelope_r1"]["assisted_layer_mask"])
        request_id = "burstgpt_longtail_eval_v2:003"
        self.assertTrue(
            controller.request_helper_envelope_is_additive(request_id, after)
        )
        controller.expand_request_helper_envelope(
            request_id, after,
            resident_component_identity_sha256=gen6.resident_component_identity_sha256,
            start_token_index=200, fraction_ppm=1_000_000,
            lease_tokens=("lease-140",), lease_reserved_until_us=400_000,
            observed_at_us=201_100,
        )
        binding = controller.request_binding(request_id)
        self.assertEqual(
            binding["helper_attachment"]["phone_session_ids"], ["HTP0", "HTP1", "HTP2"]
        )
        self.assertEqual(
            binding["helper_attachment"]["completed_phone_calls"],
            self.RECORD["attachment"]["completed_phone_calls"],
        )
        self.assertIn(
            "HELPER_EXPANDED",
            [row["kind"] for row in controller.request_helper_events(request_id)],
        )

    def test_co_helper_layers_must_be_carried_unchanged(self) -> None:
        controller, gen6 = self.attached()
        request_id = "burstgpt_longtail_eval_v2:003"
        for mask in (
            0x3FFFF,                      # drops the Pixel layers
            0x3FFFF | 0x3C0000,           # changes the co-helper layer set
            0x3F03F | PIXEL_LAYERS,       # omits HTP1's own layers
        ):
            changed = self.envelope(gen6, mask, ("HTP0", "HTP1", "HTP2"))
            self.assertFalse(
                controller.request_helper_envelope_is_additive(request_id, changed)
            )
            with self.assertRaisesRegex(
                ModelPlacementControllerError, "envelope identity is immutable"
            ):
                controller.bind_request_helper_envelope(
                    request_id, changed, observed_at_us=201_200
                )

    def test_single_phone_expansion_is_unchanged(self) -> None:
        controller, gen6 = self.attached(co_helper=0)
        request_id = "burstgpt_longtail_eval_v2:003"
        self.assertTrue(controller.request_helper_envelope_is_additive(
            request_id, self.envelope(gen6, 0x3FFFF, ("HTP0", "HTP1", "HTP2"))
        ))
        # A co-helper cannot appear in the middle of a request.
        self.assertFalse(controller.request_helper_envelope_is_additive(
            request_id,
            self.envelope(gen6, 0x3FFFF | PIXEL_LAYERS, ("HTP0", "HTP1", "HTP2")),
        ))


class ReplacementSourceTests(_ControllerCase):
    """(a) the one-session swap keeps its replacement source through the load."""

    RECORD = FIXTURE["r2_gen10_proposal"]

    def proposed(self, *, carry: bool = True):
        controller = self.controller()
        controller.carry_replacement_sources_on_update = carry
        ggg = qwen_gemma_layout(("g", "g", "g"))
        source = self.publish(controller, ggg, 100, "load-ggg")
        gqg = qwen_gemma_layout(("g", "q", "g"), current=("g", "g", "g"))
        self.assertEqual(list(gqg.changed_session_ids), self.RECORD["changed_session_ids"])
        target = controller.propose_phone_layout(
            gqg, workspace_bytes=1024, shared_compute_resource_id="op15-htp",
            shared_transport_resource_ids=("op15-functionfs", "desktop-usb-root"),
            observed_at_us=200_000,
        )
        self.assertEqual(
            [row.session_id for row in target.layout.replacement_source_identities],
            self.RECORD["proposed_replacement_sources"],
        )
        return controller, source, gqg, target

    def test_proposal_update_keeps_the_stamped_replacement_source(self) -> None:
        controller, source, gqg, target = self.proposed()
        # The same-geometry re-proposal the reprovisioner makes (unstamped).
        updated = controller.propose_phone_layout(
            gqg, workspace_bytes=1024, shared_compute_resource_id="op15-htp",
            shared_transport_resource_ids=("op15-functionfs", "desktop-usb-root"),
            observed_at_us=210_000,
        )
        self.assertEqual(updated.generation, target.generation)
        self.assertEqual(
            updated.layout.replacement_source_identities,
            (source.layout.session_identity("HTP1"),),
        )
        preparing = self.begin(controller, updated.generation, 220_000, "swap-htp1")
        self.assertEqual(preparing.state, "PREPARING")
        self.assertEqual(controller.ready_phone_layout().state, "DRAINING")
        self.assertEqual(
            preparing.layout.replacement_source_identities,
            (source.layout.session_identity("HTP1"),),
        )
        authorized = template_ops._phone_helper_authorization_layout(
            SimpleNamespace(_model_placement_controller=controller), preparing
        )
        self.assertIs(authorized, preparing.layout)

    def test_only_resident_model_reprovisioning_opts_in(self) -> None:
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        placement = scheduler._model_placement_controller
        self.assertFalse(placement.carry_replacement_sources_on_update)
        scheduler.configure_phone_resident_model_reprovisioning(
            PhoneResidentModelReprovisioningConfiguration()
        )
        self.assertTrue(placement.carry_replacement_sources_on_update)
        scheduler.configure_phone_resident_model_reprovisioning(None)
        self.assertFalse(placement.carry_replacement_sources_on_update)

    def test_without_opt_in_the_update_is_unchanged_and_the_load_authorizes(
        self,
    ) -> None:
        controller, source, gqg, target = self.proposed(carry=False)
        updated = controller.propose_phone_layout(
            gqg, workspace_bytes=1024, shared_compute_resource_id="op15-htp",
            shared_transport_resource_ids=("op15-functionfs", "desktop-usb-root"),
            observed_at_us=210_000,
        )
        # Golden behaviour: the recorded proposal is the unstamped update.
        self.assertEqual(updated.layout.replacement_source_identities, ())
        preparing = self.begin(controller, updated.generation, 220_000, "swap-htp1")
        authorized = template_ops._phone_helper_authorization_layout(
            SimpleNamespace(_model_placement_controller=controller), preparing
        )
        self.assertEqual(
            authorized.replacement_source_identities,
            (source.layout.session_identity("HTP1"),),
        )

    def test_stale_source_is_not_carried(self) -> None:
        controller, _source, gqg, target = self.proposed()
        # A session state that no longer matches the stamped source (e.g. a
        # restore with a new epoch) keeps the historical re-derivation.
        state = controller._phone_session_states["HTP1"]
        controller._phone_session_states["HTP1"] = replace(
            state, session_generation=state.session_generation + 7
        )
        updated = controller.propose_phone_layout(
            gqg, workspace_bytes=1024, shared_compute_resource_id="op15-htp",
            shared_transport_resource_ids=("op15-functionfs", "desktop-usb-root"),
            observed_at_us=210_000,
        )
        self.assertEqual(updated.generation, target.generation)
        self.assertEqual(updated.layout.replacement_source_identities, ())

    def test_in_flight_load_binds_its_own_draining_source(self) -> None:
        controller, source, _gqg, target = self.proposed()
        preparing = self.begin(controller, target.generation, 220_000, "swap-htp1")
        # The recorded PREPARING target had lost its source (PROPOSAL_UPDATED).
        self.assertEqual(self.RECORD["preparing_replacement_sources"], [])
        stripped = replace(preparing, layout=replace(
            preparing.layout,
            replacement_source_identities=(),
            replacement_source_resident_bytes_by_session={},
        ))
        view = SimpleNamespace(_model_placement_controller=controller)
        authorized = template_ops._phone_helper_authorization_layout(view, stripped)
        self.assertEqual(
            authorized.replacement_source_identities,
            (source.layout.session_identity("HTP1"),),
        )
        self.assertEqual(
            dict(authorized.replacement_source_resident_bytes_by_session),
            {"HTP1": source.layout.shards[1].resident_bytes},
        )
        # Fail closed: a DRAINING source that is not draining for this very
        # PREPARING target still rejects, now naming the blocking state.
        with self.assertRaisesRegex(
            _StalePhoneSessionAssignment,
            "replacement source is not ready: generation 1 is DRAINING "
            "while target generation 2 is PROPOSED",
        ):
            template_ops._phone_helper_authorization_layout(
                view, SimpleNamespace(
                    layout=stripped.layout,
                    generation=stripped.generation,
                    state="PROPOSED",
                ),
            )


if __name__ == "__main__":
    unittest.main()
