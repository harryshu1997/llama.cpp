from dataclasses import replace
from types import SimpleNamespace
import unittest

from research_dev.scheduler._internal.adaptive_decode import (
    ADAPTIVE_OBSERVATION_STORE_SCHEMA,
    AdaptiveDecodeController,
)
from research_dev.scheduler._internal.adaptive_decode_contracts import (
    AdaptiveDecodeConfig,
    AdaptiveDecodeError,
    AdaptiveDecodeDirective,
    AdaptiveDecodeGroupedObservation,
    AdaptiveDecodePolicy,
    AdaptiveDecodePolicyAck,
    AdaptiveDecodeRawWindowObservation,
    AdaptiveDecodeWindowBoundary,
    AdaptiveDecodeWindowReceipt,
)
from research_dev.scheduler._internal.types import canonical_sha256
from research_dev.scheduler._unified.helper_preparation import HelperPreparationMixin


ARTIFACT = "sha256:" + "1" * 64
PLAN = "sha256:" + "2" * 64
PLACEMENT = "sha256:" + "3" * 64


def policy(
    route_id: str,
    columns: int,
    layers: tuple[int, ...],
    *,
    baseline: bool = False,
) -> AdaptiveDecodePolicy:
    return AdaptiveDecodePolicy(
        route_id=route_id,
        executor_id="desktop" if baseline else "desktop-phone",
        operator_plan_sha256=PLAN,
        desktop_parent_route_id="desktop-control",
        desktop_placement_sha256=PLACEMENT,
        layer_indices=layers,
        layer_mask=sum(1 << value for value in layers),
        columns=columns,
        split_fraction_ppm=(0 if baseline else columns * 1000),
        resource_ids=(
            ("cpu", "gpu")
            if baseline else ("cpu", "gpu", "phone", "usb")
        ),
        baseline=baseline,
        predicted_latency_per_token_us=1_000,
        predicted_energy_per_token_uj=(100 if baseline else 60),
    )


class AdaptiveDecodeControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = policy("desktop-control", 0, (), baseline=True)
        self.small = policy("phone-small", 250, (2,))
        self.large = policy("phone-large", 500, (2, 3))
        self.config = AdaptiveDecodeConfig(
            minimum_remaining_tokens=4,
            minimum_window_tokens=2,
            maximum_window_tokens=2,
            maximum_probe_tokens=20,
            maximum_probe_candidates=2,
            measurement_resolution_us=1,
            transition_cost_us=1,
            transition_energy_uj=1,
            minimum_energy_saving_ppm=10_000,
            uncertainty_ppm=10_000,
            warmup_windows_per_policy=0,
        )

    def test_policy_hash_uses_canonical_payload(self) -> None:
        self.assertEqual(
            self.small.policy_hash,
            canonical_sha256(self.small._json_without_hash()),
        )

    def test_ready_poll_does_not_restart_failed_verification(self):
        controller = AdaptiveDecodeController()
        controller.start(
            request_id="request-a", ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT, planning_profile_sha256=PLAN,
            baseline=self.baseline, candidates=(self.large,), output_tokens=200,
            context_length=64, active_batch=1, deadline_us=1_000_000,
            slot_id=7, first_token_index=1, first_token_at_us=1_000,
            config=replace(self.config, maximum_probe_tokens=80,
                           maximum_probe_candidates=1, uncertainty_ppm=500_000),
            helper_available=False,
        )
        identity = dict(phone_layout_generation=1,
                        phone_layout_geometry_sha256="sha256:" + "a" * 64)
        controller.helper_ready("request-a", **identity)
        at_us = 1_000
        for stage, energy in (("initial_baseline", 100), ("candidate", 1_000)):
            session = controller.checkpoint()[1]["request-a"]
            self.assertEqual(session.stage, stage)
            token = session.target_token
            at_us += 2_000
            directive = self.record(
                controller, token, at_us, energy,
                completed_phone_calls=0 if session.current_policy.baseline else 2,
            )
            at_us += 1
            opened = self.acknowledge(controller, directive.control, token, at_us)
        self.assertEqual(controller.active_policy("request-a"), self.baseline)
        before = controller.checkpoint()[1]["request-a"]
        self.assertEqual(before.state, "EXPLOITING")
        self.assertIsNone(before.verification_policy)
        for _ in range(3):
            controller.helper_ready("request-a", **identity)
            self.assertEqual(controller.checkpoint()[1]["request-a"], before)
        continued = self.record(controller, opened.target_token_index, at_us + 2_000,
                                100, completed_phone_calls=0)
        self.assertEqual(continued.state, "EXPLOITING")
        self.assertIsNone(continued.control)
        self.assertEqual(controller.active_policy("request-a"), self.baseline)

    def test_ready_helper_rebinds_candidates_before_first_phone_use(
        self,
    ) -> None:
        controller = AdaptiveDecodeController()
        directive = controller.start(
            request_id="request-a",
            ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            component_capability_sha256=PLAN,
            baseline=self.baseline,
            candidates=(self.small,),
            output_tokens=30,
            context_length=64,
            active_batch=1,
            deadline_us=1_000_000,
            slot_id=7,
            first_token_index=1,
            first_token_at_us=1_000,
            config=self.config,
            helper_available=False,
        )
        self.assertEqual(directive.state, "PREPARING")

        component = "sha256:" + "4" * 64
        geometry = "sha256:" + "5" * 64
        controller.helper_ready(
            "request-a",
            phone_layout_generation=3,
            phone_layout_geometry_sha256=geometry,
            candidates=(self.large,),
            component_capability_sha256=component,
            ticket_policy=self.large,
        )

        snapshot = controller.snapshot("request-a")
        self.assertTrue(snapshot["helper_available"])
        self.assertEqual(snapshot["helper_layout_generation"], 3)
        self.assertEqual(
            snapshot["candidate_policy_hashes"],
            (self.large.policy_hash,),
        )
        self.assertEqual(snapshot["state"], "PROBING")

    def test_session_drain_control_is_issued_and_acked_at_boundary(self) -> None:
        controller = AdaptiveDecodeController()
        directive = controller.start(
            request_id="request-a",
            ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            component_capability_sha256=PLAN,
            baseline=self.baseline,
            candidates=(self.large,),
            output_tokens=30,
            context_length=64,
            active_batch=1,
            deadline_us=1_000_000,
            slot_id=7,
            first_token_index=1,
            first_token_at_us=1_000,
            config=self.config,
        )
        directive = self.record(
            controller, directive.target_token_index, 3_000, 100
        )
        self.assertEqual(directive.control.policy, self.large)
        opened = self.acknowledge(
            controller, directive.control, 3, 3_001
        )
        retained_mask = 1 << self.large.layer_indices[0]
        drain = controller.request_helper_session_drain(
            "request-a", retained_layer_mask=retained_mask
        )
        self.assertFalse(drain["already_applied"])
        self.assertEqual(drain["retained_layer_mask"], retained_mask)
        self.assertIsNone(controller.boundary(
            "request-a",
            slot_id=7,
            token_index=3,
            at_us=4_000,
        ))

        directive = self.record(
            controller,
            opened.target_token_index,
            5_001,
            40,
            completed_phone_calls=2,
        )
        self.assertTrue(controller.checkpoint()[1]["request-a"]
                        .records[-1].measurement_eligible)
        control = directive.control
        self.assertEqual(control.policy.layer_mask, retained_mask)
        self.assertEqual(control.to_server_json()["layer_mask"], retained_mask)
        self.assertEqual(
            controller.helper_window_bid(
                "request-a",
                requested_fraction_ppm=control.policy.split_fraction_ppm,
            )["policy_hash"],
            control.policy.policy_hash,
        )
        self.assertEqual(
            control.policy.layer_indices,
            (self.large.layer_indices[0],),
        )
        acknowledged = self.acknowledge(
            controller, control, opened.target_token_index, 5_002
        )
        self.assertEqual(
            controller.active_policy("request-a").policy_hash,
            control.policy.policy_hash,
        )
        boundary = controller.boundary(
            "request-a",
            slot_id=7,
            token_index=acknowledged.target_token_index,
            at_us=7_000,
        ).boundary
        self.assertEqual(
            boundary.applied_ack.policy_hash,
            control.policy.policy_hash,
        )

    def test_session_drain_shortens_window_without_qualifying_sample(self):
        controller = AdaptiveDecodeController()
        directive = self.start(controller)
        directive = self.record(
            controller, directive.target_token_index, 3_000, 100
        )
        old_policy = directive.control.policy
        self.assertEqual(old_policy, self.large)
        opened = self.acknowledge(controller, directive.control, 3, 3_001)
        self.assertEqual(opened.target_token_index, 5)
        self.assertIsNone(controller.boundary(
            "request-a", slot_id=7, token_index=4, at_us=4_000,
        ))
        before = controller.checkpoint()[1]["request-a"]
        retained_mask = 1 << old_policy.layer_indices[0]
        drain = controller.request_helper_session_drain(
            "request-a", retained_layer_mask=retained_mask
        )
        self.assertEqual(controller.request_helper_session_drain(
            "request-a", retained_layer_mask=retained_mask
        ), drain)

        shortened = controller.boundary(
            "request-a", slot_id=7, token_index=4, at_us=4_000,
        )
        self.assertIsNotNone(shortened)
        self.assertEqual(shortened.boundary.policy, old_policy)
        self.assertEqual(shortened.boundary.token_start, 3)
        self.assertEqual(shortened.boundary.token_end, 4)
        self.assertIsNone(shortened.control)
        directive = self.record(
            controller, 4, 4_000, 40, completed_phone_calls=2
        )
        session = controller.checkpoint()[1]["request-a"]
        receipt = session.records[-1]
        self.assertEqual(receipt.policy, old_policy)
        self.assertEqual(receipt.token_count, 1)
        self.assertEqual(receipt.fleet_energy_uj_by_domain, {"fleet": 40})
        self.assertEqual(receipt.completed_phone_calls, 2)
        self.assertEqual(receipt.phone_compute_us, 10)
        self.assertEqual(receipt.usb_transfer_us, 5)
        self.assertEqual(receipt.rpc_us, 2)
        self.assertFalse(receipt.measurement_eligible)
        self.assertFalse(receipt.energy_measurement_eligible)
        self.assertEqual(session.records[:-1], before.records)
        self.assertEqual(
            session.warmup_windows_seen_by_policy,
            before.warmup_windows_seen_by_policy,
        )
        control = directive.control
        self.assertEqual(control.policy.layer_mask, retained_mask)
        self.assertEqual(control.policy.columns, old_policy.columns)
        self.assertEqual(control.policy.split_fraction_ppm,
                         old_policy.split_fraction_ppm)
        self.assertEqual(control.to_server_json()["layer_mask"], retained_mask)
        self.assertEqual(session.transition_policy, old_policy)
        self.assertIsNone(controller.boundary(
            "request-a", slot_id=7, token_index=5, at_us=5_000,
        ))
        self.assertEqual(controller.snapshot("request-a")["record_count"], 2)
        controller.acknowledge(
            "request-a",
            AdaptiveDecodePolicyAck(
                request_id="request-a", slot_id=7,
                plan_generation=control.plan_generation,
                applied_token_index=5, applied_at_us=5_001,
                policy_hash=control.policy.policy_hash,
            ),
            transition_observation=AdaptiveDecodeRawWindowObservation(
                fleet_energy_uj_by_domain={"fleet": 42},
                phone_compute_us=10, usb_transfer_us=5, rpc_us=2,
                exposed_tail_us=1, output_valid=True,
                evidence_ids=("synthetic:drain-ack",),
                energy_boundary_id="synthetic-fleet",
                energy_attribution_kind="isolated",
                completed_phone_calls=2, completed_phone_input_rows=8,
            ),
        )
        transition = controller.checkpoint()[1]["request-a"].records[-1]
        self.assertEqual(transition.policy, old_policy)
        self.assertEqual((transition.token_start, transition.token_end), (4, 5))
        self.assertEqual(transition.fleet_energy_uj_by_domain, {"fleet": 42})
        self.assertEqual(transition.completed_phone_calls, 2)
        self.assertFalse(transition.measurement_eligible)
        self.assertEqual(controller.active_policy("request-a"), control.policy)
        self.assertTrue(controller.request_helper_session_drain(
            "request-a", retained_layer_mask=retained_mask
        )["already_applied"])

    def test_pending_session_drain_waits_for_inflight_control_ack(self):
        controller = AdaptiveDecodeController()
        directive = self.start(controller)
        directive = self.record(
            controller, directive.target_token_index, 3_000, 100
        )
        old_control = directive.control
        retained_mask = 1 << old_control.policy.layer_indices[0]
        drain = controller.request_helper_session_drain(
            "request-a", retained_layer_mask=retained_mask
        )
        self.assertIsNone(controller.boundary(
            "request-a", slot_id=7, token_index=3, at_us=3_001,
        ))
        self.assertEqual(controller.snapshot("request-a")[
            "awaiting_control"
        ], old_control.to_json())
        directive = self.acknowledge(controller, old_control, 3, 3_002)
        self.assertEqual(directive.control.policy.policy_hash,
                         drain["policy_hash"])
        self.assertEqual(directive.control.policy.layer_mask, retained_mask)
        self.assertGreater(directive.control.plan_generation,
                           old_control.plan_generation)
        self.assertEqual(
            controller.checkpoint()[1]["request-a"].transition_policy,
            old_control.policy,
        )
        self.acknowledge(controller, directive.control, 3, 3_003)
        self.assertEqual(controller.active_policy("request-a"),
                         directive.control.policy)

    def test_immediate_helper_expansion_preserves_short_generation_proof(self):
        from research_dev.scheduler.adapters.llama_server import (
            LlamaServerFfnCall, LlamaServerFfnCallContext, ManagedLlamaServer,
        )

        controller = AdaptiveDecodeController()
        directive = self.start(controller)
        directive = self.record(controller, directive.target_token_index, 3_000, 100)
        old_control = directive.control
        expanded = tuple(replace(
            row, operator_plan_sha256="sha256:" + "4" * 64,
            layer_indices=(2, 3, 4), layer_mask=28,
        ) for row in (self.small, self.large))
        controller.helper_rebound(
            "request-a", phone_layout_generation=2,
            phone_layout_geometry_sha256="sha256:" + "5" * 64,
            candidates=expanded, component_capability_sha256=PLAN,
            ticket_policy=None, helper_evidence_state="LEARNING",
        )
        old_ack = AdaptiveDecodePolicyAck(
            request_id="request-a", slot_id=7,
            plan_generation=old_control.plan_generation,
            applied_token_index=3, applied_at_us=3_001,
            policy_hash=old_control.policy.policy_hash,
        )
        directive = controller.acknowledge("request-a", old_ack)
        self.assertIsNotNone(directive.control)
        self.assertEqual(directive.control.policy, expanded[1])
        self.assertEqual(controller.snapshot("request-a")["record_count"], 1)
        checkpoint = controller.checkpoint()
        deferred = controller.defer_control(
            "request-a", directive.control, "synthetic:busy", at_us=3_002,
        )
        boundary = controller.boundary(
            "request-a", slot_id=7, token_index=deferred.target_token_index,
            at_us=5_002,
        ).boundary
        self.assertEqual(boundary.policy, old_control.policy)
        self.assertEqual(boundary.applied_ack, old_ack)
        controller.restore(checkpoint)
        controller.acknowledge(
            "request-a", AdaptiveDecodePolicyAck(
                request_id="request-a", slot_id=7,
                plan_generation=directive.control.plan_generation,
                applied_token_index=4, applied_at_us=4_001,
                policy_hash=directive.control.policy.policy_hash,
            ), transition_observation=AdaptiveDecodeRawWindowObservation(
                fleet_energy_uj_by_domain={"fleet": 40}, phone_compute_us=10,
                usb_transfer_us=5, rpc_us=2, exposed_tail_us=1, output_valid=True,
                evidence_ids=("synthetic:successive-controls",),
                energy_boundary_id="synthetic-fleet", energy_attribution_kind="isolated",
                completed_phone_calls=2, completed_phone_input_rows=2,
            ),
        )
        records = controller.checkpoint()[1]["request-a"].records
        transition = records[-1]
        self.assertEqual((transition.token_start, transition.token_end), (3, 4))
        self.assertEqual(transition.policy, old_control.policy)
        self.assertEqual(transition.applied_ack, old_ack)
        self.assertFalse(transition.measurement_eligible)
        self.assertFalse(transition.energy_measurement_eligible)
        calls = tuple(LlamaServerFfnCall(
            request_id=index + 1, layer=layer, columns=old_control.policy.columns,
            tokens=1, payload_bytes=64,
            contexts=(LlamaServerFfnCallContext("request-a", 7, 1,
                                               old_ack.plan_generation),),
        ) for index, layer in enumerate(old_control.policy.layer_indices))
        server = object.__new__(ManagedLlamaServer)
        arguments = (
            tuple((call.layer, call.columns) for call in calls),
            SimpleNamespace(windows=records, final_policy_ack=None),
            SimpleNamespace(request_id="request-a"),
            SimpleNamespace(embedding_length=32),
            {PLAN: SimpleNamespace(layer_indices=self.large.layer_indices, columns=500)},
            {"request-a"},
        )
        invalid, bindings = server._verify_single_request_adaptive_calls(calls, *arguments)
        self.assertFalse(invalid)
        self.assertEqual(bindings, {call.request_id: PLAN for call in calls})
        stale = (replace(calls[0], contexts=(replace(
            calls[0].contexts[0], plan_generation=old_ack.plan_generation + 10,
        ),)), *calls[1:])
        self.assertTrue(server._verify_single_request_adaptive_calls(stale, *arguments)[0])

    def start(
        self,
        controller: AdaptiveDecodeController,
        *,
        deadline_us: int = 1_000_000,
    ):
        return controller.start(
            request_id="request-a",
            ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            baseline=self.baseline,
            candidates=(self.small, self.large),
            output_tokens=30,
            context_length=64,
            active_batch=1,
            deadline_us=deadline_us,
            slot_id=7,
            first_token_index=1,
            first_token_at_us=1_000,
            config=self.config,
        )

    @staticmethod
    def acknowledge(controller, control, token, at_us):
        return controller.acknowledge(
            "request-a",
            AdaptiveDecodePolicyAck(
                request_id="request-a",
                slot_id=7,
                plan_generation=control.plan_generation,
                applied_token_index=token,
                applied_at_us=at_us,
                policy_hash=control.policy.policy_hash,
            ),
        )

    def record(
        self,
        controller,
        token,
        at_us,
        energy_per_token,
        *,
        active_batch=None,
        next_active_batch=None,
        membership_changed=False,
        execution_context_available=True,
        compatible_batch_change=False,
        attribution_kind="isolated",
        completed_phone_calls=None,
        failure_reason=None,
        external_activity_sha256=None,
    ):
        directive = controller.boundary(
            "request-a", slot_id=7, token_index=token, at_us=at_us
        )
        self.assertIsNotNone(directive)
        boundary = directive.boundary
        self.assertIsNotNone(boundary)
        phone = not boundary.policy.baseline
        return controller.record_window(
            "request-a",
            boundary,
            AdaptiveDecodeRawWindowObservation(
                fleet_energy_uj_by_domain={
                    "fleet": energy_per_token * boundary.token_count
                },
                phone_compute_us=10 if phone else 0,
                usb_transfer_us=5 if phone else 0,
                rpc_us=2 if phone else 0,
                exposed_tail_us=1 if phone else 0,
                output_valid=True,
                evidence_ids=("synthetic:window",),
                energy_boundary_id="synthetic-fleet",
                energy_attribution_kind=attribution_kind,
                failure_reason=failure_reason,
                usb_upload_bytes=20 if phone else 0,
                usb_download_bytes=20 if phone else 0,
                desktop_compute_us=12 if phone else 20,
                useful_overlap_us=8 if phone else 0,
                request_queue_delay_us=3,
                protected_interference_us=0,
                active_batch=active_batch,
                next_active_batch=next_active_batch,
                membership_changed=membership_changed,
                execution_context_available=execution_context_available,
                completed_phone_calls=completed_phone_calls,
                completed_phone_input_rows=(
                    None
                    if completed_phone_calls is None else
                    completed_phone_calls * 4
                ),
                external_activity_sha256=external_activity_sha256,
            ),
            compatible_batch_change=compatible_batch_change,
        )

    def _external_activity_controller(self):
        controller = AdaptiveDecodeController()
        directive = controller.start(
            request_id="request-a", ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT, planning_profile_sha256=PLAN,
            baseline=self.baseline, candidates=(self.large,), output_tokens=200,
            context_length=64, active_batch=1, deadline_us=10_000_000,
            slot_id=7, first_token_index=1, first_token_at_us=1_000,
            config=replace(self.config, maximum_probe_tokens=80),
        )
        return controller, directive

    def _step_external(self, controller, directive, at_us, external, *,
                       assisted_energy, baseline_energy=100):
        session = controller.checkpoint()[1]["request-a"]
        current = session.current_policy
        energy = baseline_energy if current is None or current.baseline else assisted_energy
        directive = self.record(
            controller, directive.target_token_index, at_us, energy,
            completed_phone_calls=None if current is None or current.baseline else 2,
            external_activity_sha256=external,
        )
        if directive.control is not None:
            # Acknowledge at the boundary itself so synthetic windows keep equal
            # per-token latency under the zero-slack latency gate.
            session = controller.checkpoint()[1]["request-a"]
            directive = self.acknowledge(
                controller, directive.control,
                directive.target_token_index or session.transition_start_token, at_us,
            )
        return directive

    def test_first_external_activity_identity_is_not_a_context_change(self):
        controller, directive = self._external_activity_controller()
        quiet = "sha256:" + "a" * 64
        at_us = 1_000
        for _ in range(4):
            at_us += 2_000
            directive = self._step_external(controller, directive, at_us, quiet, assisted_energy=50)
        session = controller.checkpoint()[1]["request-a"]
        self.assertEqual(session.external_activity_sha256, quiet)
        self.assertEqual(session.context_record_start, 0)
        self.assertTrue(all(row.measurement_eligible for row in session.records))
        self.assertFalse(any(row.external_activity_changed for row in session.records))
        self.assertEqual(session.state, "EXPLOITING")
        self.assertEqual(session.incumbent_policy, self.large)
        snapshot = controller.snapshot("request-a")
        self.assertEqual(snapshot["external_activity_sha256"], quiet)
        receipt = session.records[-1]
        self.assertEqual(receipt.external_activity_sha256, quiet)
        self.assertEqual(AdaptiveDecodeWindowReceipt.from_json(receipt.to_json()), receipt)
        self.assertNotIn("external_activity_changed", receipt.to_json())

    def test_external_activity_change_monitors_incumbent_in_a_new_context(self):
        controller, directive = self._external_activity_controller()
        quiet = "sha256:" + "a" * 64
        co_run = "sha256:" + "b" * 64
        at_us = 1_000
        for _ in range(4):
            at_us += 2_000
            directive = self._step_external(controller, directive, at_us, quiet, assisted_energy=50)
        session = controller.checkpoint()[1]["request-a"]
        self.assertEqual(session.incumbent_policy, self.large)
        records_before = len(session.records)
        at_us += 2_000
        # The assisted window that first sees other desktop work is mixed evidence.
        directive = self._step_external(controller, directive, at_us, co_run, assisted_energy=120)
        session = controller.checkpoint()[1]["request-a"]
        crossing = session.records[records_before]
        self.assertFalse(crossing.measurement_eligible)
        self.assertTrue(crossing.external_activity_changed)
        self.assertEqual(crossing.external_activity_sha256, co_run)
        self.assertEqual(AdaptiveDecodeWindowReceipt.from_json(crossing.to_json()), crossing)
        self.assertEqual(session.context_record_start, records_before + 1)
        self.assertEqual(session.external_activity_sha256, co_run)
        self.assertIsNone(session.incumbent_policy)
        self.assertEqual(session.state, "PROBING")
        self.assertTrue(session.operational_verification)
        self.assertEqual(session.verification_policy, self.large)
        self.assertEqual(controller.snapshot("request-a")["zero_assistance_reason"],
                         "EXTERNAL_ACTIVITY_CHANGED")
        prior = controller.snapshot("request-a")["context_monitor_prior"]
        self.assertEqual(prior["source_policy_hash"], self.large.policy_hash)
        # Only measurements taken in the new context decide the incumbent's fate.
        for _ in range(4):
            at_us += 2_000
            directive = self._step_external(controller, directive, at_us, co_run, assisted_energy=120)
        session = controller.checkpoint()[1]["request-a"]
        self.assertEqual(session.state, "EXPLOITING")
        self.assertTrue(session.current_policy.baseline)
        self.assertEqual(session.eliminated_policy_reasons,
                         {self.large.policy_hash: "CURRENT_PAIR_NOT_IMPROVED"})
        self.assertEqual(session.zero_assistance_reason, "MEASURED_REJECTION")
        # Quiet-context evidence (energy 50) no longer counts; only the co-run
        # measurements of the same policy do.
        current = controller._current_valid_records(session, self.large, operational=True)
        self.assertTrue(current)
        self.assertTrue(all(
            row.window_index > crossing.window_index
            and row.external_activity_sha256 == co_run
            and row.energy_per_token_uj == 120
            for row in current))
        self.assertEqual(session.probe_tokens, sum(
            row.token_count for row in session.records
            if row.window_role == "exploration" and not row.policy.baseline))

    def test_external_activity_end_clears_eliminations_and_recovers_incumbent(self):
        controller, directive = self._external_activity_controller()
        quiet = "sha256:" + "a" * 64
        co_run = "sha256:" + "b" * 64
        at_us = 1_000
        for _ in range(4):
            at_us += 2_000
            directive = self._step_external(controller, directive, at_us, quiet, assisted_energy=50)
        for _ in range(6):
            at_us += 2_000
            directive = self._step_external(controller, directive, at_us, co_run, assisted_energy=120)
        session = controller.checkpoint()[1]["request-a"]
        self.assertTrue(session.current_policy.baseline)
        self.assertIn(self.large.policy_hash, session.eliminated_policy_reasons)
        records_before = len(session.records)
        at_us += 2_000
        directive = self._step_external(controller, directive, at_us, quiet, assisted_energy=50)
        session = controller.checkpoint()[1]["request-a"]
        self.assertEqual(session.context_record_start, records_before + 1)
        self.assertFalse(session.records[records_before].measurement_eligible)
        self.assertTrue(session.records[records_before].external_activity_changed)
        self.assertEqual(session.eliminated_policy_reasons, {})
        self.assertEqual(session.state, "PROBING")
        self.assertEqual(session.stage, "initial_baseline")
        self.assertEqual(session.zero_assistance_reason, "EXTERNAL_ACTIVITY_CHANGED")
        # A fresh baseline window in the quiet context re-opens the probe.
        at_us += 2_000
        directive = self._step_external(controller, directive, at_us, quiet, assisted_energy=50)
        session = controller.checkpoint()[1]["request-a"]
        self.assertEqual(session.current_policy, self.large)
        for _ in range(3):
            at_us += 2_000
            directive = self._step_external(controller, directive, at_us, quiet, assisted_energy=50)
        session = controller.checkpoint()[1]["request-a"]
        self.assertEqual(session.state, "EXPLOITING")
        self.assertEqual(session.incumbent_policy, self.large)
        self.assertEqual(session.current_policy, self.large)

    def test_unknown_external_activity_never_resets_context(self):
        controller, directive = self._external_activity_controller()
        quiet = "sha256:" + "a" * 64
        at_us = 1_000
        for _ in range(4):
            at_us += 2_000
            directive = self._step_external(controller, directive, at_us, quiet, assisted_energy=50)
        for _ in range(2):
            at_us += 2_000
            directive = self._step_external(controller, directive, at_us, None, assisted_energy=50)
        session = controller.checkpoint()[1]["request-a"]
        self.assertEqual(session.context_record_start, 0)
        self.assertEqual(session.external_activity_sha256, quiet)
        self.assertEqual(session.incumbent_policy, self.large)
        self.assertIsNone(session.records[-1].external_activity_sha256)
        with self.assertRaisesRegex(AdaptiveDecodeError, "external activity identity"):
            AdaptiveDecodeRawWindowObservation(
                fleet_energy_uj_by_domain={"fleet": 1}, phone_compute_us=0, usb_transfer_us=0,
                rpc_us=0, exposed_tail_us=0, output_valid=True, evidence_ids=("x",),
                energy_boundary_id="synthetic-fleet", energy_attribution_kind="isolated",
                external_activity_sha256="peer-ticket",
            )

    def test_assistance_summaries_expose_measured_loss_bounds(self):
        controller, directive = self._external_activity_controller()
        quiet = "sha256:" + "a" * 64
        at_us = 1_000
        rows = controller.assistance_summaries()
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["assisted"])
        self.assertIsNone(rows[0]["loss_upper_per_token_uj"])
        for _ in range(4):
            at_us += 2_000
            directive = self._step_external(controller, directive, at_us, quiet, assisted_energy=50)
        session = controller.checkpoint()[1]["request-a"]
        self.assertEqual(session.incumbent_policy, self.large)
        row = controller.assistance_summaries()[0]
        self.assertEqual(row["request_id"], "request-a")
        self.assertTrue(row["assisted"])
        self.assertEqual(row["assisted_fraction_ppm"], self.large.split_fraction_ppm)
        baseline = controller._bounds(session, session.baseline, operational=True)
        assisted = controller._bounds(session, self.large, operational=True)
        self.assertEqual(row["loss_upper_per_token_uj"], baseline[2] - assisted[1])
        self.assertGreater(row["loss_upper_per_token_uj"], 0)
        self.assertEqual(row["assisted_latency_per_token_us"], assisted[3])
        self.assertEqual(row["remaining_tokens"], 200 - session.records[-1].token_end)
        self.assertEqual(row["external_activity_sha256"], quiet)
        # Losing the helper means there is no assistance left to lose.
        controller.helper_unavailable("request-a")
        self.assertFalse(controller.assistance_summaries()[0]["assisted"])

    def test_external_activity_change_midway_through_verification_restarts_pairing(self):
        controller, directive = self._external_activity_controller()
        quiet = "sha256:" + "a" * 64
        co_run = "sha256:" + "b" * 64
        later = "sha256:" + "c" * 64
        at_us = 1_000
        for _ in range(4):
            at_us += 2_000
            directive = self._step_external(controller, directive, at_us, quiet, assisted_energy=50)
        at_us += 2_000
        directive = self._step_external(controller, directive, at_us, co_run, assisted_energy=120)
        session = controller.checkpoint()[1]["request-a"]
        self.assertTrue(session.operational_verification)
        self.assertEqual(session.stage, "cached_candidate")
        # The context moves again before the pair completes: the partial pair
        # is evidence of nothing and a new bounded monitor starts.
        at_us += 2_000
        directive = self._step_external(controller, directive, at_us, later, assisted_energy=120)
        session = controller.checkpoint()[1]["request-a"]
        self.assertTrue(session.records[-1].external_activity_changed)
        self.assertFalse(session.records[-1].measurement_eligible)
        self.assertEqual(session.context_record_start, len(session.records))
        self.assertEqual(session.external_activity_sha256, later)
        self.assertEqual(session.eliminated_policy_reasons, {})
        self.assertEqual(session.state, "PROBING")
        self.assertIsNone(session.incumbent_policy)
        # Verification attempts are a request-wide limit, not refilled per context.
        self.assertEqual(session.verification_attempts, 2)
        for _ in range(5):
            at_us += 2_000
            directive = self._step_external(controller, directive, at_us, later, assisted_energy=50)
        session = controller.checkpoint()[1]["request-a"]
        self.assertEqual(session.state, "EXPLOITING")
        self.assertEqual(session.incumbent_policy, self.large)

    def test_external_activity_change_with_insufficient_remaining_tokens_stays_baseline(self):
        controller, directive = self._external_activity_controller()
        quiet = "sha256:" + "a" * 64
        co_run = "sha256:" + "b" * 64
        at_us = 1_000
        for _ in range(4):
            at_us += 2_000
            directive = self._step_external(controller, directive, at_us, quiet, assisted_energy=50)
        for _ in range(6):
            at_us += 2_000
            directive = self._step_external(controller, directive, at_us, co_run, assisted_energy=120)
        session = controller.checkpoint()[1]["request-a"]
        self.assertTrue(session.current_policy.baseline)
        # Move to the last affordable boundaries before the 200-token request ends.
        while directive.target_token_index is not None and directive.target_token_index < 196:
            at_us += 2_000
            directive = self._step_external(controller, directive, at_us, co_run, assisted_energy=120)
        session = controller.checkpoint()[1]["request-a"]
        self.assertTrue(session.current_policy.baseline)
        at_us += 2_000
        directive = self._step_external(controller, directive, at_us, quiet, assisted_energy=50)
        session = controller.checkpoint()[1]["request-a"]
        self.assertEqual(session.eliminated_policy_reasons, {})
        self.assertTrue(session.current_policy is None or session.current_policy.baseline)
        self.assertIsNone(session.incumbent_policy)
        while directive.target_token_index is not None:
            at_us += 2_000
            directive = self._step_external(controller, directive, at_us, quiet, assisted_energy=50)
            self.assertTrue(controller.active_policy("request-a").baseline)
        session = controller.checkpoint()[1]["request-a"]
        self.assertEqual(directive.reason, "TERMINAL_WINDOW_RECORDED")
        self.assertEqual(session.state, "EXPLOITING")
        self.assertTrue(all(row.policy.baseline
                            for row in session.records[session.context_record_start:]))

    def test_external_activity_change_without_helper_never_issues_phone_control(self):
        controller, directive = self._external_activity_controller()
        quiet = "sha256:" + "a" * 64
        co_run = "sha256:" + "b" * 64
        at_us = 1_000
        for _ in range(4):
            at_us += 2_000
            directive = self._step_external(controller, directive, at_us, quiet, assisted_energy=50)
        self.assertEqual(controller.checkpoint()[1]["request-a"].incumbent_policy, self.large)
        # Session loss: the helper generation is stale, so assistance stops first.
        controller.helper_unavailable("request-a")
        at_us += 2_000
        directive = self._step_external(controller, directive, at_us, quiet, assisted_energy=50)
        self.assertTrue(controller.active_policy("request-a").baseline)
        for _ in range(3):
            at_us += 2_000
            directive = self._step_external(controller, directive, at_us, co_run, assisted_energy=50)
            self.assertTrue(controller.active_policy("request-a").baseline)
            self.assertIsNone(directive.control)
        session = controller.checkpoint()[1]["request-a"]
        self.assertFalse(session.helper_available)
        self.assertEqual(session.zero_assistance_reason, "PHONE_HELPER_UNAVAILABLE")
        self.assertEqual(session.state, "PREPARING")
        self.assertEqual(session.external_activity_sha256, co_run)
        # A fresh generation makes the request eligible again only at a boundary.
        controller.helper_ready(
            "request-a", phone_layout_generation=2,
            phone_layout_geometry_sha256="sha256:" + "9" * 64,
        )
        for _ in range(4):
            at_us += 2_000
            directive = self._step_external(controller, directive, at_us, co_run, assisted_energy=50)
        session = controller.checkpoint()[1]["request-a"]
        self.assertEqual(session.helper_layout_generation, 2)
        self.assertEqual(session.state, "EXPLOITING")
        self.assertEqual(session.incumbent_policy, self.large)

    def test_maintenance_drain_precedes_external_context_monitoring(self):
        controller, directive = self._external_activity_controller()
        quiet = "sha256:" + "a" * 64
        co_run = "sha256:" + "b" * 64
        at_us = 1_000
        for _ in range(4):
            at_us += 2_000
            directive = self._step_external(controller, directive, at_us, quiet, assisted_energy=50)
        session = controller.checkpoint()[1]["request-a"]
        self.assertEqual(session.current_policy, self.large)
        retained_mask = 1 << self.large.layer_indices[0]
        drain = controller.request_helper_session_drain(
            "request-a", retained_layer_mask=retained_mask,
        )
        self.assertFalse(drain["already_applied"])
        at_us += 2_000
        directive = self.record(
            controller, directive.target_token_index, at_us, 120,
            completed_phone_calls=2, external_activity_sha256=co_run,
        )
        session = controller.checkpoint()[1]["request-a"]
        self.assertTrue(session.records[-1].external_activity_changed)
        self.assertIsNotNone(directive.control)
        self.assertEqual(directive.control.policy.layer_mask, retained_mask)
        self.assertIsNone(session.pending_session_drain_policy)
        self.assertEqual(session.context_record_start, len(session.records))
        self.assertIsNone(session.incumbent_policy)
        directive = self.acknowledge(
            controller, directive.control, session.transition_start_token, at_us + 1,
        )
        self.assertEqual(controller.active_policy("request-a").layer_mask, retained_mask)
        at_us += 2_000
        directive = self.record(
            controller, directive.target_token_index, at_us, 120,
            completed_phone_calls=2, external_activity_sha256=co_run,
        )
        session = controller.checkpoint()[1]["request-a"]
        self.assertFalse(session.records[-1].external_activity_changed)
        self.assertEqual(session.external_activity_sha256, co_run)

    def test_learning_helper_uses_configured_high_value_probe_order(self):
        controller = AdaptiveDecodeController()
        directive = controller.start(
            request_id="request-a",
            ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            baseline=self.baseline,
            candidates=(self.small, self.large),
            output_tokens=30,
            context_length=64,
            active_batch=1,
            deadline_us=1_000_000,
            slot_id=7,
            first_token_index=1,
            first_token_at_us=1_000,
            config=self.config,
            helper_evidence_state="LEARNING",
        )
        snapshot = controller.snapshot("request-a")
        self.assertEqual(snapshot["helper_evidence_state"], "LEARNING")
        self.assertEqual(
            snapshot["probe_fractions_ppm"], (500_000, 250_000)
        )

        directive = self.record(
            controller, directive.target_token_index, 3_000, 100
        )
        self.assertEqual(
            directive.control.policy.split_fraction_ppm, 500_000
        )
        opened = self.acknowledge(
            controller, directive.control, 3, 3_001
        )
        directive = self.record(
            controller,
            opened.target_token_index,
            4_501,
            80,
            completed_phone_calls=2,
        )
        self.assertEqual(
            directive.control.policy.split_fraction_ppm, 250_000
        )

    def test_learning_starts_at_full_assistance_then_steps_down(self):
        controller = AdaptiveDecodeController()
        seventy_five = policy("phone-seventy-five", 750, (2, 3))
        full = policy("phone-full", 1000, (2, 3))

        controller.start(
            request_id="request-a",
            ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            baseline=self.baseline,
            candidates=(self.small, self.large, seventy_five, full),
            output_tokens=30,
            context_length=64,
            active_batch=1,
            deadline_us=1_000_000,
            slot_id=7,
            first_token_index=1,
            first_token_at_us=1_000,
            config=replace(self.config, maximum_probe_candidates=4),
            helper_evidence_state="LEARNING",
        )

        self.assertEqual(
            controller.snapshot("request-a")["probe_fractions_ppm"],
            (1_000_000, 750_000, 500_000, 250_000),
        )

    def test_learning_with_a_compatible_baseline_probes_full_first(self):
        controller = AdaptiveDecodeController()
        directive = controller.start(
            request_id="request-a",
            ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            baseline=self.baseline,
            candidates=(self.small, self.large),
            output_tokens=6,
            context_length=64,
            active_batch=1,
            deadline_us=1_000_000,
            slot_id=7,
            first_token_index=0,
            first_token_at_us=1_000,
            config=self.config,
            helper_available=False,
            helper_evidence_state="LEARNING",
        )
        at_us = 1_000
        while directive.target_token_index < 6:
            at_us += 2_000
            directive = self.record(
                controller, directive.target_token_index, at_us, 100
            )
        at_us += 2_000
        self.record(controller, 6, at_us, 100)
        controller.complete("request-a", "COMPLETED")

        full = policy("phone-full", 1000, (2, 3))
        seventy_five = policy("phone-seventy-five", 750, (2, 3))
        directive = controller.start(
            request_id="request-b",
            ticket_id="request-b:attempt:0",
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            baseline=self.baseline,
            candidates=(self.small, self.large, seventy_five, full),
            output_tokens=30,
            context_length=64,
            active_batch=1,
            deadline_us=1_000_000,
            slot_id=8,
            first_token_index=0,
            first_token_at_us=20_000,
            config=replace(self.config, maximum_probe_candidates=4),
            helper_evidence_state="LEARNING",
        )

        self.assertIsNotNone(directive.control)
        self.assertEqual(
            directive.control.policy.split_fraction_ppm, 1_000_000
        )

    def test_learning_reuses_diagnostic_baseline_only_for_operational_selection(self):
        _, source = self._run(40)
        previous = "0" * 64
        windows = []
        for row in source.windows:
            window = replace(
                row,
                evidence_ids=("ASSUMED_4P5W", "synthetic:window"),
                energy_attribution_kind="diagnostic",
                previous_record_sha256=previous,
            )
            windows.append(window)
            previous = window.record_sha256.removeprefix("sha256:")
        historical = replace(source, windows=tuple(windows))
        body = {
            "groups": [historical.to_json()],
            "schema": ADAPTIVE_OBSERVATION_STORE_SCHEMA,
        }
        for allowed in (True, False):
            with self.subTest(operational_assumed_power=allowed):
                controller = AdaptiveDecodeController()
                controller.load_observations({
                    **body, "store_sha256": canonical_sha256(body),
                })
                directive = controller.start(
                    request_id="learning-b", ticket_id="learning-b:attempt:0",
                    model_artifact_sha256=ARTIFACT,
                    planning_profile_sha256=PLAN,
                    baseline=self.baseline, candidates=(self.small, self.large),
                    output_tokens=30, context_length=64, active_batch=1,
                    deadline_us=1_000_000, slot_id=8,
                    first_token_index=1, first_token_at_us=1_000,
                    config=replace(
                        self.config,
                        allow_assumed_phone_power_for_operational_selection=allowed,
                    ),
                    helper_evidence_state="LEARNING",
                )
                self.assertEqual(directive.state, "PROBING")
                if not allowed:
                    self.assertIsNone(directive.control)
                    self.assertTrue(controller.active_policy("learning-b").baseline)
                    continue
                control = directive.control
                self.assertEqual(control.policy, self.large)
                opened = controller.acknowledge(
                    "learning-b", AdaptiveDecodePolicyAck(
                        request_id="learning-b", slot_id=8,
                        plan_generation=control.plan_generation,
                        applied_token_index=1, applied_at_us=1_001,
                        policy_hash=control.policy.policy_hash,
                    ),
                )
                boundary = controller.boundary(
                    "learning-b", slot_id=8,
                    token_index=opened.target_token_index, at_us=2_501,
                ).boundary
                controller.record_window(
                    "learning-b", boundary, AdaptiveDecodeRawWindowObservation(
                        fleet_energy_uj_by_domain={"fleet": 40 * boundary.token_count},
                        phone_compute_us=10, usb_transfer_us=5, rpc_us=2,
                        exposed_tail_us=1, output_valid=True,
                        evidence_ids=("ASSUMED_4P5W", "synthetic:window"),
                        energy_boundary_id="synthetic-fleet",
                        energy_attribution_kind="diagnostic",
                        completed_phone_calls=2,
                        completed_phone_input_rows=2,
                    ),
                )
                self.assertNotIn(
                    self.large.policy_hash,
                    controller.snapshot("learning-b")["eliminated_policy_reasons"],
                )
                session = controller.checkpoint()[1]["learning-b"]
                self.assertFalse(controller._valid_records(session, self.baseline))
                self.assertTrue(controller._valid_records(
                    session, self.baseline, operational=True,
                ))
                self.assertIsNone(controller._cached_verification_policy(session))

    def _diagnostic_winner_history(self):
        _, source = self._run(40)
        previous, windows = "0" * 64, []
        for row in source.windows:
            energy = 100 if row.policy.baseline else (40 if row.policy == self.small else 140)
            window = replace(
                row, evidence_ids=("ASSUMED_4P5W", "synthetic:window"),
                energy_attribution_kind="diagnostic",
                fleet_energy_uj_by_domain={"fleet": energy * row.token_count},
                finished_at_us=(row.finished_at_us if row.policy.baseline
                                else row.started_at_us + 500 * row.token_count),
                latency_per_token_us=1_000 if row.policy.baseline else 500,
                completed_phone_calls=0 if row.policy.baseline else row.token_count,
                completed_phone_input_rows=0 if row.policy.baseline else row.token_count,
                previous_record_sha256=previous,
            )
            windows.append(window)
            previous = window.record_sha256.removeprefix("sha256:")
        return replace(source, windows=tuple(windows), final_policy=self.small,
                       helper_layout_geometry_sha256="sha256:" + "a" * 64)

    def _start_diagnostic_winner(self, history, **overrides):
        controller = AdaptiveDecodeController()
        body = {"groups": [history.to_json()], "schema": ADAPTIVE_OBSERVATION_STORE_SCHEMA}
        controller.load_observations({**body, "store_sha256": canonical_sha256(body)})
        args = dict(
            request_id="request-b", ticket_id="request-b:attempt:0",
            model_artifact_sha256=ARTIFACT, planning_profile_sha256=PLAN,
            baseline=self.baseline, candidates=(self.small, self.large), output_tokens=30,
            context_length=64, active_batch=1, deadline_us=1_000_000,
            slot_id=8, first_token_index=1, first_token_at_us=1_000,
            config=replace(self.config, allow_assumed_phone_power_for_operational_selection=True),
            helper_evidence_state="LEARNING", helper_layout_generation=1,
            helper_layout_geometry_sha256=history.helper_layout_geometry_sha256,
        )
        args.update(overrides)
        return controller, controller.start(**args)

    def _verify_diagnostic_winner(self, controller, directive, phone_energy):
        at_us = 1_000
        token = 1
        for _ in range(8):
            if directive.control is not None:
                control = directive.control
                at_us += 1
                directive = controller.acknowledge("request-b", AdaptiveDecodePolicyAck(
                    request_id="request-b", slot_id=8, plan_generation=control.plan_generation,
                    applied_token_index=token, applied_at_us=at_us,
                    policy_hash=control.policy.policy_hash,
                ))
            if directive.state == "EXPLOITING":
                return
            policy_now = controller.active_policy("request-b")
            at_us += 2_000 if policy_now.baseline else 1_000
            token = directive.target_token_index
            boundary = controller.boundary("request-b", slot_id=8,
                                          token_index=token, at_us=at_us).boundary
            directive = controller.record_window("request-b", boundary, AdaptiveDecodeRawWindowObservation(
                fleet_energy_uj_by_domain={"fleet": (100 if policy_now.baseline else phone_energy) * boundary.token_count},
                phone_compute_us=0, usb_transfer_us=0, rpc_us=0, exposed_tail_us=0,
                output_valid=True, evidence_ids=("ASSUMED_4P5W", "synthetic:window"),
                energy_boundary_id="synthetic-fleet", energy_attribution_kind="diagnostic",
                completed_phone_calls=0 if policy_now.baseline else boundary.token_count,
                completed_phone_input_rows=0 if policy_now.baseline else boundary.token_count,
            ))
        self.fail("bounded winner verification did not converge")

    def test_diagnostic_winner_reuse_verifies_one_pair_without_full_sweep(self):
        controller, directive = self._start_diagnostic_winner(self._diagnostic_winner_history())
        self.assertEqual(directive.control.policy, self.small)
        self.assertEqual(directive.state, "PROBING")
        self._verify_diagnostic_winner(controller, directive, 40)
        session = controller.checkpoint()[1]["request-b"]
        self.assertEqual(session.current_policy, self.small)
        self.assertEqual([row.policy for row in session.records], [self.small, self.baseline])
        self.assertFalse(any(row.energy_measurement_eligible for row in session.records))
        self.assertIsNone(session.cached_winner)
        self.assertIsNone(controller._cached_verification_policy(session))
        self.assertEqual(session.probe_candidates, [self.small])

    def test_diagnostic_winner_fresh_negative_pair_overrides_positive_history(self):
        controller, directive = self._start_diagnostic_winner(self._diagnostic_winner_history())
        self._verify_diagnostic_winner(controller, directive, 200)
        session = controller.checkpoint()[1]["request-b"]
        self.assertEqual(session.current_policy, self.baseline)
        self.assertEqual(len(session.records), 2)
        self.assertEqual(controller.snapshot("request-b")["verification"]["outcome"], "REJECTED")
        controller.helper_ready("request-b", phone_layout_generation=1,
                                phone_layout_geometry_sha256="sha256:" + "a" * 64)
        self.assertEqual(controller.checkpoint()[1]["request-b"], session)

    def test_diagnostic_winner_requires_exact_parent_layout_component_shape_and_permission(self):
        source = self._diagnostic_winner_history()
        for change in (
            {"helper_layout_geometry_sha256": "sha256:" + "b" * 64},
            {"helper_layout_geometry_sha256": None, "helper_layout_generation": None},
            {"model_artifact_sha256": "sha256:" + "b" * 64},
            {"planning_profile_sha256": "sha256:" + "b" * 64},
            {"active_batch": 2},
            {"config": self.config},
            {"baseline": replace(self.baseline, desktop_placement_sha256="sha256:" + "b" * 64),
             "candidates": tuple(replace(row, desktop_placement_sha256="sha256:" + "b" * 64)
                                 for row in (self.small, self.large))},
        ):
            with self.subTest(change=change):
                controller, _ = self._start_diagnostic_winner(source, **change)
                self.assertFalse(controller.checkpoint()[1]["request-b"].operational_verification)

    def test_diagnostic_winner_in_neighbouring_prompt_bucket_is_a_starting_point(self):
        """Physical v15: a verified Qwen winner measured at a 300-token prompt
        was ignored by every request whose prompt fell into another
        power-of-two bucket, so those requests probed from scratch or gave up.
        The nearest bucket's operational winner now seeds the bounded paired
        verification (a starting point, never qualified history); the exact
        bucket is still preferred when both exist."""
        source = self._diagnostic_winner_history()
        controller, directive = self._start_diagnostic_winner(source, context_length=256)
        session = controller.checkpoint()[1]["request-b"]
        self.assertTrue(session.operational_verification)
        self.assertEqual(session.verification_policy, self.small)
        self.assertIsNone(session.cached_winner)
        self.assertEqual(session.probe_candidates, [self.small])
        # Far buckets are still reached only through the nearest-first order.
        far, _ = self._start_diagnostic_winner(source, context_length=4096)
        self.assertTrue(far.checkpoint()[1]["request-b"].operational_verification)

    def test_diagnostic_winner_late_ready_uses_current_baseline_then_one_candidate(self):
        controller, directive = self._start_diagnostic_winner(
            self._diagnostic_winner_history(), helper_available=False,
            helper_layout_generation=None, helper_layout_geometry_sha256=None,
        )
        self.assertTrue(controller.active_policy("request-b").baseline)
        controller.helper_ready("request-b", phone_layout_generation=1,
                                phone_layout_geometry_sha256="sha256:" + "a" * 64)
        session = controller.checkpoint()[1]["request-b"]
        self.assertTrue(session.operational_verification)
        self.assertEqual(session.stage, "verification_baseline")
        self._verify_diagnostic_winner(controller, directive, 40)
        self.assertEqual(controller.active_policy("request-b"), self.small)
        self.assertEqual(len(controller.checkpoint()[1]["request-b"].records), 2)
        with self.assertRaisesRegex(AdaptiveDecodeError, "generation changed"):
            controller.helper_ready("request-b", phone_layout_generation=2,
                                    phone_layout_geometry_sha256="sha256:" + "a" * 64)

    def test_diagnostic_winner_too_short_request_stays_on_desktop(self):
        controller, directive = self._start_diagnostic_winner(self._diagnostic_winner_history(), output_tokens=2)
        self.assertIsNone(directive.control)
        self.assertTrue(controller.active_policy("request-b").baseline)

    def _record_reuse(self, controller, directive, token, at_us, *, valid=True):
        if directive.control is not None:
            control = directive.control
            directive = controller.acknowledge("request-b", AdaptiveDecodePolicyAck(
                request_id="request-b", slot_id=8, plan_generation=control.plan_generation,
                applied_token_index=token, applied_at_us=at_us, policy_hash=control.policy.policy_hash,
            ))
        policy_now = controller.active_policy("request-b")
        target = directive.target_token_index
        at_us += (target - token) * (1_000 if policy_now.baseline else 500)
        boundary = controller.boundary("request-b", slot_id=8, token_index=target, at_us=at_us).boundary
        observation = AdaptiveDecodeRawWindowObservation(
            fleet_energy_uj_by_domain={"fleet": (100 if policy_now.baseline else 40) * boundary.token_count},
            phone_compute_us=0, usb_transfer_us=0, rpc_us=0, exposed_tail_us=0,
            output_valid=True, evidence_ids=(("ASSUMED_4P5W", "synthetic:window")
                                             if valid else ("MEASUREMENT_UNAVAILABLE",)),
            energy_boundary_id="synthetic-fleet", energy_attribution_kind="diagnostic",
            completed_phone_calls=0 if policy_now.baseline else boundary.token_count,
            completed_phone_input_rows=0 if policy_now.baseline else boundary.token_count,
        )
        return controller.record_window("request-b", boundary, observation), target, at_us

    def test_unaffordable_pair_runs_the_prior_under_monitoring(self):
        """Physical v15: 171 Qwen tokens in requests of 26-71 tokens stayed at
        0% with VERIFICATION_INCOMPLETE/COMPLETE_PAIR_BUDGET although the same
        winner had just been verified on the same layout. When the paired
        verification is unaffordable but the winner itself can be measured,
        the request now starts at the winner under per-window monitoring: no
        verification attempt is spent, nothing is promoted to incumbent."""
        controller, directive = self._start_diagnostic_winner(
            self._diagnostic_winner_history(), output_tokens=12,
            config=replace(self.config, warmup_windows_per_policy=1,
                           allow_assumed_phone_power_for_operational_selection=True))
        session = controller.checkpoint()[1]["request-b"]
        self.assertFalse(controller._can_probe(session, 1, 1_000))
        self.assertIsNotNone(directive.control)
        self.assertEqual(directive.control.policy, self.small)
        self.assertEqual(directive.reason, "VERIFICATION_MONITORING")
        self.assertEqual(session.stage, "prior_monitor")
        self.assertEqual(session.prior_monitor_policy, self.small)
        self.assertFalse(session.operational_verification)
        self.assertEqual(session.verification_attempts, 0)
        self.assertIsNone(session.incumbent_policy)
        self.assertEqual(session.verification_reason, "PRIOR_UNDER_MONITOR")
        self.assertEqual(
            controller.helper_attachment_opportunity("request-b", token_index=1, at_us=1_000),
            "ELIGIBLE",
        )

    def test_too_short_for_any_measured_window_still_defers_the_pair(self):
        controller, directive = self._start_diagnostic_winner(
            self._diagnostic_winner_history(), output_tokens=4,
            config=replace(self.config, warmup_windows_per_policy=1,
                           allow_assumed_phone_power_for_operational_selection=True))
        session = controller.checkpoint()[1]["request-b"]
        # 3 remaining tokens cannot hold a warm-up window plus a measured one.
        self.assertIsNone(directive.control)
        self.assertEqual(directive.reason, "VERIFICATION_INCOMPLETE")
        self.assertEqual(session.verification_attempts, 0)
        self.assertEqual(session.verification_reason, "COMPLETE_PAIR_BUDGET")
        self.assertIsNone(session.prior_monitor_policy)

    def _monitor_prior(self, controller, directive, phone_energy, windows):
        at_us, token = 1_000, 1
        for _ in range(windows):
            if directive.control is not None:
                control = directive.control
                at_us += 1
                directive = controller.acknowledge("request-b", AdaptiveDecodePolicyAck(
                    request_id="request-b", slot_id=8, plan_generation=control.plan_generation,
                    applied_token_index=token, applied_at_us=at_us,
                    policy_hash=control.policy.policy_hash,
                ))
            policy_now = controller.active_policy("request-b")
            target = directive.target_token_index
            at_us += (target - token) * (1_000 if policy_now.baseline else 500)
            token = target
            boundary = controller.boundary("request-b", slot_id=8, token_index=token, at_us=at_us).boundary
            directive = controller.record_window("request-b", boundary, AdaptiveDecodeRawWindowObservation(
                fleet_energy_uj_by_domain={"fleet": (100 if policy_now.baseline else phone_energy) * boundary.token_count},
                phone_compute_us=0, usb_transfer_us=0, rpc_us=0, exposed_tail_us=0,
                output_valid=True, evidence_ids=("ASSUMED_4P5W", "synthetic:window"),
                energy_boundary_id="synthetic-fleet", energy_attribution_kind="diagnostic",
                completed_phone_calls=0 if policy_now.baseline else boundary.token_count,
                completed_phone_input_rows=0 if policy_now.baseline else boundary.token_count,
            ))
            if token >= 30:
                break
        return directive

    def test_prior_monitor_keeps_the_winner_while_its_windows_beat_the_baseline(self):
        controller, directive = self._start_diagnostic_winner(
            self._diagnostic_winner_history(), output_tokens=30,
            config=replace(self.config, warmup_windows_per_policy=1, minimum_remaining_tokens=100,
                           allow_assumed_phone_power_for_operational_selection=True))
        self.assertEqual(directive.reason, "VERIFICATION_MONITORING")
        directive = self._monitor_prior(controller, directive, phone_energy=40, windows=6)
        session = controller.checkpoint()[1]["request-b"]
        self.assertEqual(session.stage, "prior_monitor")
        self.assertEqual(controller.active_policy("request-b"), self.small)
        self.assertGreaterEqual(session.prior_monitor_windows, 2)
        self.assertIsNone(session.incumbent_policy)
        self.assertNotIn(self.small.policy_hash, session.eliminated_policy_reasons)
        self.assertTrue(all(not row.policy.baseline for row in session.records[1:]))

    def test_prior_monitor_drops_to_desktop_when_the_winner_stops_paying(self):
        controller, directive = self._start_diagnostic_winner(
            self._diagnostic_winner_history(), output_tokens=30,
            config=replace(self.config, warmup_windows_per_policy=1, minimum_remaining_tokens=100,
                           allow_assumed_phone_power_for_operational_selection=True))
        directive = self._monitor_prior(controller, directive, phone_energy=140, windows=3)
        session = controller.checkpoint()[1]["request-b"]
        self.assertEqual(session.eliminated_policy_reasons.get(self.small.policy_hash), "PRIOR_MONITOR_NOT_IMPROVED")
        self.assertEqual(session.verification_outcome, "REJECTED")
        self.assertEqual(session.verification_reason, "PRIOR_MONITOR_REJECTED")
        self.assertIsNone(session.prior_monitor_policy)
        self.assertEqual(session.stage, "initial_baseline")
        self.assertTrue(directive.control is not None and directive.control.policy.baseline
                        or controller.active_policy("request-b").baseline)
        self.assertNotEqual(
            controller.helper_attachment_opportunity("request-b", token_index=20, at_us=50_000),
            "ELIGIBLE",
        )

    def test_helper_loss_during_prior_monitor_returns_to_desktop_and_clears_prior(self):
        controller, directive = self._start_diagnostic_winner(
            self._diagnostic_winner_history(), output_tokens=30,
            config=replace(self.config, warmup_windows_per_policy=1, minimum_remaining_tokens=100,
                           allow_assumed_phone_power_for_operational_selection=True))
        directive = self._monitor_prior(controller, directive, phone_energy=40, windows=2)
        controller.helper_unavailable("request-b")
        session = controller.checkpoint()[1]["request-b"]
        self.assertIsNone(session.prior_monitor_policy)
        self.assertFalse(session.helper_available)
        self.assertEqual(session.zero_assistance_reason, "PHONE_HELPER_UNAVAILABLE")

    def test_reserved_pair_survives_shrinking_deadline_window_allowance(self):
        source = self._diagnostic_winner_history()
        previous, rows, time_us = "0" * 64, [], 1_000
        for row in source.windows:
            latency = 612_141 if row.policy.baseline else 500_000
            changed = replace(row, started_at_us=time_us,
                              finished_at_us=time_us + row.token_count * latency,
                              latency_per_token_us=latency, applied_ack=None,
                              previous_record_sha256=previous)
            rows.append(changed)
            previous = changed.record_sha256.removeprefix("sha256:")
            time_us = changed.finished_at_us + 1
        source = replace(source, windows=tuple(rows))
        controller, directive = self._start_diagnostic_winner(
            source, output_tokens=133, deadline_us=380_000_000,
            first_token_at_us=361_987_162,
            config=AdaptiveDecodeConfig(minimum_energy_saving_ppm=10_000,
                transition_energy_uj=self.config.transition_energy_uj,
                maximum_latency_ppm=1_250_000, allow_assumed_phone_power_for_operational_selection=True))
        self.assertEqual(directive.reason, "VERIFICATION_RESERVED")
        session = controller._sessions["request-b"]
        # Reproduce the measured token-11 budget without injecting qualification.
        self.assertEqual((380_000_000 - 366_825_772) * 150_000 // 1_000_000, 1_976_134)
        self.assertEqual(controller._token_latency_us(session, session.baseline) * 4, 2_448_564)
        self.assertTrue(controller._can_probe(session, 11, 366_825_772))
        self.assertGreater(session.verification_budget["deadline_us"], 366_825_772)
        control = directive.control
        directive = controller.acknowledge("request-b", AdaptiveDecodePolicyAck(
            request_id="request-b", slot_id=8, plan_generation=control.plan_generation,
            applied_token_index=1, applied_at_us=361_987_163, policy_hash=control.policy.policy_hash))
        time_us = 361_987_163
        for _ in range(2):
            time_us += 2_000_000
            boundary = controller.boundary("request-b", slot_id=8,
                token_index=directive.target_token_index, at_us=time_us).boundary
            directive = controller.record_window("request-b", boundary, AdaptiveDecodeRawWindowObservation(
                fleet_energy_uj_by_domain={"fleet": 160}, phone_compute_us=0,
                usb_transfer_us=0, rpc_us=0, exposed_tail_us=0, output_valid=True,
                evidence_ids=("ASSUMED_4P5W",), energy_boundary_id="synthetic-fleet",
                energy_attribution_kind="diagnostic", completed_phone_calls=4,
                completed_phone_input_rows=4))
        self.assertEqual(directive.control.policy, self.baseline)
        self.assertEqual(directive.state, "PROBING")
        self.assertEqual(controller.snapshot("request-b")["verification"]["outcome"], "RESERVED")

    def test_incomplete_pair_retries_once_then_exploits_valid_current_pair(self):
        controller, directive = self._start_diagnostic_winner(
            self._diagnostic_winner_history(), output_tokens=200)
        token, at_us = 1, 1_000
        for _ in range(2):
            directive, token, at_us = self._record_reuse(controller, directive, token, at_us, valid=False)
        self.assertEqual(directive.reason, "VERIFICATION_INCOMPLETE")
        self.assertEqual(controller.snapshot("request-b")["verification"]["attempts"], 1)
        for _ in range(6):
            directive, token, at_us = self._record_reuse(controller, directive, token, at_us)
            if directive.reason == "VERIFICATION_VERIFIED":
                break
        self.assertEqual(directive.reason, "VERIFICATION_VERIFIED")
        snapshot = controller.snapshot("request-b")["verification"]
        self.assertEqual(snapshot["attempts"], 2)
        if directive.control is not None:
            self.assertEqual(directive.control.policy, self.small)
        else:
            self.assertEqual(controller.active_policy("request-b"), self.small)

    def test_persistent_incomplete_evidence_has_bounded_retry_and_no_rejection(self):
        controller, directive = self._start_diagnostic_winner(
            self._diagnostic_winner_history(), output_tokens=200)
        token, at_us = 1, 1_000
        for _ in range(12):
            directive, token, at_us = self._record_reuse(controller, directive, token, at_us, valid=False)
        snapshot = controller.snapshot("request-b")
        self.assertEqual(snapshot["verification"]["attempts"], 2)
        self.assertEqual(snapshot["verification"]["outcome"], "INCOMPLETE")
        self.assertEqual(snapshot["verification"]["reason"], "RETRY_LIMIT")
        self.assertEqual(snapshot["eliminated_policy_reasons"], {})
        self.assertEqual(controller.active_policy("request-b"), self.baseline)

    def test_verification_reservation_expiry_is_incomplete_and_does_not_wait(self):
        controller, directive = self._start_diagnostic_winner(
            self._diagnostic_winner_history(), output_tokens=200)
        control = directive.control
        deadline = controller.snapshot("request-b")["verification"]["budget"]["deadline_us"]
        directive = controller.acknowledge("request-b", AdaptiveDecodePolicyAck(
            request_id="request-b", slot_id=8, plan_generation=control.plan_generation,
            applied_token_index=1, applied_at_us=1_000, policy_hash=control.policy.policy_hash))
        directive, _, _ = self._record_reuse(controller, directive, 1, deadline)
        self.assertEqual(directive.control.policy, self.baseline)
        self.assertEqual(directive.reason, "VERIFICATION_INCOMPLETE")
        self.assertEqual(controller.snapshot("request-b")["verification"]["reason"], "RESERVATION_EXHAUSTED")

    def test_verification_control_failure_is_incomplete_not_retried(self):
        controller, directive = self._start_diagnostic_winner(self._diagnostic_winner_history())
        failed = controller.control_failed("request-b", directive.control, "PROOF_REJECTED", at_us=1_001)
        self.assertEqual(failed.control.policy, self.baseline)
        self.assertEqual(failed.reason, "VERIFICATION_INCOMPLETE")
        self.assertFalse(controller.checkpoint()[1]["request-b"].operational_verification)

    def test_session_drain_cancels_pair_without_restoring_full_mask(self):
        # The diagnostic winner covers one layer, so use an exact superset fixture.
        history = replace(self._diagnostic_winner_history(), final_policy=self.large)
        rows, previous = [], "0" * 64
        for row in history.windows:
            changed = replace(row, policy=self.large if row.policy == self.small else row.policy,
                              applied_ack=None, previous_record_sha256=previous)
            rows.append(changed)
            previous = changed.record_sha256.removeprefix("sha256:")
        controller, directive = self._start_diagnostic_winner(replace(history, windows=tuple(rows)))
        control = directive.control
        self.assertEqual(control.policy, self.large)
        directive = controller.acknowledge("request-b", AdaptiveDecodePolicyAck(
            request_id="request-b", slot_id=8, plan_generation=control.plan_generation,
            applied_token_index=1, applied_at_us=1_000, policy_hash=control.policy.policy_hash))
        controller.request_helper_session_drain("request-b", retained_layer_mask=1 << 2)
        directive, token, at_us = self._record_reuse(controller, directive, 1, 1_000)
        self.assertEqual(directive.control.policy.layer_mask, 1 << 2)
        self.assertFalse(controller.checkpoint()[1]["request-b"].operational_verification)
        directive, _, _ = self._record_reuse(controller, directive, token, at_us)
        self.assertIsNone(directive.control)
        self.assertEqual(controller.active_policy("request-b").layer_mask, 1 << 2)

    def test_learning_uses_highest_supported_configured_probe(self):
        controller = AdaptiveDecodeController()
        three_eighths = policy("phone-three-eighths", 375, (2,))
        directive = controller.start(
            request_id="request-a",
            ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            baseline=self.baseline,
            candidates=(three_eighths, self.large),
            output_tokens=30,
            context_length=64,
            active_batch=1,
            deadline_us=1_000_000,
            slot_id=7,
            first_token_index=1,
            first_token_at_us=1_000,
            config=replace(self.config, minimum_remaining_tokens=24),
            helper_evidence_state="LEARNING",
        )
        self.assertEqual(
            controller.snapshot("request-a")["probe_fractions_ppm"],
            (500_000, 375_000),
        )

        directive = self.record(
            controller, directive.target_token_index, 3_000, 100
        )

        self.assertEqual(
            directive.control.policy.split_fraction_ppm, 500_000
        )

    def test_learning_ignores_structural_ticket_until_helper_ready(self):
        controller = AdaptiveDecodeController()
        structural = policy("phone-structural", 750, (2, 3, 4))
        directive = controller.start(
            request_id="request-a",
            ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            baseline=self.baseline,
            candidates=(self.small, self.large),
            output_tokens=30,
            context_length=64,
            active_batch=1,
            deadline_us=1_000_000,
            slot_id=7,
            first_token_index=1,
            first_token_at_us=1_000,
            config=self.config,
            ticket_policy=structural,
            helper_available=False,
            helper_evidence_state="LEARNING",
        )

        self.assertEqual(directive.state, "PREPARING")
        self.assertTrue(
            controller.active_policy("request-a").baseline
        )
        self.assertEqual(
            controller.active_policy("request-a").split_fraction_ppm, 0
        )
        controller.helper_ready(
            "request-a",
            phone_layout_generation=2,
            phone_layout_geometry_sha256="sha256:" + "5" * 64,
        )
        directive = self.record(
            controller, directive.target_token_index, 3_000, 100
        )
        self.assertEqual(
            directive.control.policy.split_fraction_ppm, 500_000
        )

    def test_learning_helper_returns_to_desktop_without_paired_gain(self):
        controller = AdaptiveDecodeController()
        directive = controller.start(
            request_id="request-a",
            ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            baseline=self.baseline,
            candidates=(self.small, self.large),
            output_tokens=30,
            context_length=64,
            active_batch=1,
            deadline_us=1_000_000,
            slot_id=7,
            first_token_index=1,
            first_token_at_us=1_000,
            config=self.config,
            helper_evidence_state="LEARNING",
        )
        directive = self.record(
            controller, directive.target_token_index, 3_000, 100
        )
        opened = self.acknowledge(
            controller, directive.control, 3, 3_001
        )
        directive = self.record(
            controller,
            opened.target_token_index,
            5_001,
            110,
            completed_phone_calls=2,
        )
        self.assertEqual(directive.control.policy, self.small)
        opened = self.acknowledge(
            controller,
            directive.control,
            opened.target_token_index,
            5_002,
        )
        directive = self.record(
            controller,
            opened.target_token_index,
            7_002,
            110,
            completed_phone_calls=2,
        )
        self.assertTrue(directive.control.policy.baseline)
        snapshot = controller.snapshot("request-a")
        self.assertEqual(
            snapshot["eliminated_policy_reasons"][
                self.large.policy_hash
            ],
            "LEARNING_NO_PAIRED_IMPROVEMENT",
        )
        self.assertEqual(
            snapshot["eliminated_policy_reasons"][
                self.small.policy_hash
            ],
            "LEARNING_NO_PAIRED_IMPROVEMENT",
        )

    def test_learning_helper_failure_immediately_requests_zero_fraction(self):
        controller = AdaptiveDecodeController()
        directive = controller.start(
            request_id="request-a",
            ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            baseline=self.baseline,
            candidates=(self.small,),
            output_tokens=30,
            context_length=64,
            active_batch=1,
            deadline_us=1_000_000,
            slot_id=7,
            first_token_index=1,
            first_token_at_us=1_000,
            config=self.config,
            helper_evidence_state="LEARNING",
        )
        directive = self.record(
            controller, directive.target_token_index, 3_000, 100
        )
        opened = self.acknowledge(
            controller, directive.control, 3, 3_001
        )
        directive = self.record(
            controller,
            opened.target_token_index,
            4_501,
            80,
            completed_phone_calls=1,
            failure_reason="synthetic-helper-failure",
        )
        self.assertTrue(directive.control.policy.baseline)
        self.assertEqual(directive.control.policy.split_fraction_ppm, 0)
        self.assertEqual(
            controller.snapshot("request-a")["state"], "RECOVERING"
        )

    def test_live_batch_change_restarts_from_a_paired_baseline(self):
        controller = AdaptiveDecodeController()
        directive = self.start(controller)
        directive = self.record(
            controller, directive.target_token_index, 3_000, 100
        )
        opened = self.acknowledge(
            controller, directive.control, 3, 3_001
        )

        directive = self.record(
            controller,
            opened.target_token_index,
            5_001,
            40,
            active_batch=4,
        )

        self.assertIsNotNone(directive.control)
        self.assertTrue(directive.control.policy.baseline)
        snapshot = controller.snapshot("request-a")
        self.assertEqual(snapshot["active_batch"], 4)
        self.assertEqual(snapshot["state"], "PROBING")

    def test_cohort_shrink_rekeys_after_preserving_window_batch(self):
        controller = AdaptiveDecodeController()
        directive = controller.start(
            request_id="request-a",
            ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            baseline=self.baseline,
            candidates=(self.small, self.large),
            output_tokens=30,
            context_length=64,
            active_batch=2,
            deadline_us=1_000_000,
            slot_id=7,
            first_token_index=1,
            first_token_at_us=1_000,
            config=self.config,
        )
        boundary = controller.boundary(
            "request-a",
            slot_id=7,
            token_index=directive.target_token_index,
            at_us=3_000,
        ).boundary
        controller.record_window(
            "request-a",
            boundary,
            AdaptiveDecodeRawWindowObservation(
                fleet_energy_uj_by_domain={"fleet": 400},
                phone_compute_us=0,
                usb_transfer_us=0,
                rpc_us=0,
                exposed_tail_us=0,
                output_valid=True,
                evidence_ids=("synthetic:cohort-shrink",),
                energy_boundary_id="synthetic-fleet",
                energy_attribution_kind="isolated",
                active_batch=2,
                next_active_batch=1,
                accounting_token_count=4,
                cohort_id="synthetic-cohort",
                cohort_member_request_ids=("request-a", "request-b"),
                energy_owner_request_id="request-a",
            ),
        )

        checkpoint = controller.checkpoint()
        receipt = checkpoint[1]["request-a"].records[-1]
        self.assertEqual(receipt.active_batch, 2)
        self.assertEqual(receipt.next_active_batch, 1)
        self.assertEqual(
            AdaptiveDecodeWindowReceipt.from_json(receipt.to_json()),
            receipt,
        )
        self.assertEqual(controller.snapshot("request-a")["active_batch"], 1)

    def test_token_boundary_control_rejects_stale_acknowledgement(self):
        controller = AdaptiveDecodeController()
        directive = self.start(controller)
        directive = self.record(
            controller, directive.target_token_index, 3_000, 100
        )
        control = directive.control
        with self.assertRaisesRegex(
            AdaptiveDecodeError, "acknowledgement differs"
        ):
            controller.acknowledge(
                "request-a",
                AdaptiveDecodePolicyAck(
                    request_id="request-a",
                    slot_id=7,
                    plan_generation=control.plan_generation + 1,
                    applied_token_index=3,
                    applied_at_us=3_100,
                    policy_hash=control.policy.policy_hash,
                ),
            )
        opened = self.acknowledge(controller, control, 3, 3_100)
        self.assertEqual(opened.target_token_index, 5)
        self.assertEqual(
            controller.snapshot("request-a")["current_policy_hash"],
            self.large.policy_hash,
        )

        self.assertIsNone(controller.boundary(
            "request-a",
            slot_id=7,
            token_index=2,
            at_us=3_200,
        ))
        self.assertIsNotNone(controller.boundary(
            "request-a",
            slot_id=7,
            token_index=5,
            at_us=5_100,
        ))

    def test_short_decode_keeps_initial_policy_without_late_control(self):
        controller = AdaptiveDecodeController()
        directive = controller.start(
            request_id="request-a",
            ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            baseline=self.baseline,
            candidates=(self.small, self.large),
            output_tokens=3,
            context_length=64,
            active_batch=1,
            deadline_us=1_000_000,
            slot_id=7,
            first_token_index=1,
            first_token_at_us=1_000,
            config=self.config,
            ticket_policy=self.large,
        )

        self.assertIsNone(directive.control)
        self.assertEqual(directive.target_token_index, 3)
        snapshot = controller.snapshot("request-a")
        self.assertEqual(snapshot["state"], "EXPLOITING")
        self.assertEqual(
            snapshot["current_policy_hash"], self.baseline.policy_hash
        )

    def test_tail_can_seal_at_an_acknowledged_control_boundary(self):
        for acknowledged_token in (29, 30):
            with self.subTest(acknowledged_token=acknowledged_token):
                controller = AdaptiveDecodeController()
                self.start(controller)
                directive = self.record(controller, 3, 3_000, 100)
                acknowledgement = AdaptiveDecodePolicyAck(
                    request_id="request-a", slot_id=7,
                    plan_generation=directive.control.plan_generation,
                    applied_token_index=acknowledged_token,
                    applied_at_us=30_000,
                    policy_hash=directive.control.policy.policy_hash,
                )
                controller.acknowledge(
                    "request-a", acknowledgement,
                    AdaptiveDecodeRawWindowObservation(
                        fleet_energy_uj_by_domain={"fleet": 2700},
                        phone_compute_us=0, usb_transfer_us=0,
                        rpc_us=0, exposed_tail_us=0, output_valid=True,
                        evidence_ids=("physical:control-transition-ack",),
                        energy_boundary_id="synthetic-fleet",
                        energy_attribution_kind="diagnostic",
                    ),
                )
                controller.seal_tail(
                    "request-a", slot_id=7, token_index=acknowledged_token,
                    reason="server_release_guard",
                )
                preview = controller.preview_completion("request-a")
                self.assertEqual(preview.windows[-1].token_end, acknowledged_token)
                self.assertEqual(preview.unmeasured_tail_tokens, 30 - acknowledged_token)
                self.assertEqual(preview.final_policy, directive.control.policy)
                self.assertEqual(
                    preview.final_policy_ack,
                    acknowledgement if acknowledged_token < 30 else None,
                )
                self.assertEqual(
                    AdaptiveDecodeGroupedObservation.from_json(preview.to_json()),
                    preview,
                )
                if acknowledged_token < 30:
                    for invalid_ack in (
                        replace(acknowledgement, request_id="another-request"),
                        replace(acknowledgement, slot_id=8),
                        replace(acknowledgement, applied_token_index=28),
                        replace(acknowledgement, applied_at_us=29_999),
                        replace(acknowledgement, policy_hash=self.baseline.policy_hash),
                    ):
                        with self.assertRaisesRegex(
                            AdaptiveDecodeError, "final acknowledgement differs"
                        ):
                            replace(preview, final_policy_ack=invalid_ack)
                    tampered = preview.to_json()
                    tampered["final_policy_ack"]["plan_generation"] += 1
                    with self.assertRaisesRegex(AdaptiveDecodeError, "grouped hash differs"):
                        AdaptiveDecodeGroupedObservation.from_json(tampered)
                self.assertTrue(all(row.failure_reason is None for row in preview.windows))
                self.assertEqual(controller.complete("request-a", "COMPLETED"), preview)

    def test_tail_cannot_seal_an_unacknowledged_empty_window(self):
        controller = AdaptiveDecodeController()
        self.start(controller)
        with self.assertRaisesRegex(AdaptiveDecodeError, "tail seal is invalid"):
            controller.seal_tail(
                "request-a", slot_id=7, token_index=1,
                reason="server_release_guard",
            )

    def test_final_token_seal_has_no_unmeasured_tail(self):
        for stale in (False, True):
            with self.subTest(stale=stale):
                controller = AdaptiveDecodeController()
                self.start(controller)
                controller.seal_tail(
                    "request-a", slot_id=7, token_index=30,
                    reason="server_release_guard",
                )
                if stale:
                    boundary = controller.boundary(
                        "request-a", slot_id=7, token_index=30,
                        at_us=30_000, terminal=True,
                    ).boundary
                    controller.discard_stale_window(
                        "request-a", boundary,
                        AdaptiveDecodeRawWindowObservation(
                            fleet_energy_uj_by_domain={"fleet": 100},
                            phone_compute_us=0, usb_transfer_us=0,
                            rpc_us=0, exposed_tail_us=0,
                            output_valid=True,
                            evidence_ids=("synthetic:stale-slot",),
                            energy_boundary_id="synthetic-fleet",
                            energy_attribution_kind="diagnostic",
                            failure_reason="stale_slot_stats_discarded",
                        ),
                        "stale_slot_stats_discarded",
                    )
                else:
                    self.record(controller, 30, 30_000, 100)
                preview = controller.preview_completion("request-a")
                self.assertEqual(preview.windows[-1].token_end, 30)
                self.assertEqual(preview.unmeasured_tail_tokens, 0)
                self.assertIsNone(preview.unmeasured_tail_reason)
                if stale:
                    self.assertFalse(preview.windows[-1].measurement_eligible)
                    self.assertEqual(
                        preview.windows[-1].failure_reason,
                        "stale_slot_stats_discarded",
                    )
                self.assertEqual(
                    controller.complete("request-a", "COMPLETED"), preview
                )

    def test_stale_short_tail_window_is_diagnostic_and_completes(self):
        controller = AdaptiveDecodeController()
        controller.start(
            request_id="request-a",
            ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            baseline=self.baseline,
            candidates=(self.small, self.large),
            output_tokens=3,
            context_length=64,
            active_batch=1,
            deadline_us=1_000_000,
            slot_id=7,
            first_token_index=1,
            first_token_at_us=1_000,
            config=self.config,
            ticket_policy=self.large,
        )
        controller.seal_tail(
            "request-a",
            slot_id=7,
            token_index=2,
            reason="server_release_guard",
        )
        directive = controller.boundary(
            "request-a",
            slot_id=7,
            token_index=2,
            at_us=2_000,
            terminal=True,
        )
        self.assertIsInstance(directive, AdaptiveDecodeDirective)
        self.assertIsInstance(
            directive.boundary, AdaptiveDecodeWindowBoundary
        )
        discarded = controller.discard_stale_window(
            "request-a",
            directive.boundary,
            AdaptiveDecodeRawWindowObservation(
                fleet_energy_uj_by_domain={"fleet": 100},
                phone_compute_us=0,
                usb_transfer_us=0,
                rpc_us=0,
                exposed_tail_us=0,
                output_valid=True,
                evidence_ids=("synthetic:stale-slot",),
                energy_boundary_id="synthetic-fleet",
                energy_attribution_kind="diagnostic",
                failure_reason="stale_slot_stats_discarded",
                completed_phone_calls=0,
                completed_phone_input_rows=0,
            ),
            "stale_slot_stats_discarded",
        )

        self.assertEqual(
            discarded.reason, "STALE_SLOT_DIRECTIVE_DISCARDED"
        )
        preview = controller.preview_completion("request-a")
        self.assertEqual(len(preview.windows), 1)
        self.assertFalse(preview.windows[0].measurement_eligible)
        self.assertEqual(
            preview.windows[0].failure_reason,
            "stale_slot_stats_discarded",
        )
        self.assertEqual(preview.final_policy, self.baseline)
        self.assertEqual(preview.unmeasured_tail_tokens, 1)
        completed = controller.complete("request-a", "COMPLETED")
        self.assertEqual(completed.terminal_status, "COMPLETED")

    def test_released_phone_tail_preserves_measured_prefix_and_ack(self):
        controller = AdaptiveDecodeController()
        directive = self.start(controller)
        at_us = 1_000
        while directive.target_token_index <= 27:
            token = directive.target_token_index
            active = controller.active_policy("request-a")
            at_us += 2_000
            directive = self.record(
                controller, token, at_us, 100 if active.baseline else 30,
                completed_phone_calls=0 if active.baseline else 4,
            )
            while directive.control is not None:
                at_us += 1
                directive = self.acknowledge(controller, directive.control, token, at_us)
        before = controller.checkpoint()[1]["request-a"]
        self.assertEqual(before.state, "EXPLOITING")
        self.assertFalse(before.current_policy.baseline)
        self.assertEqual(before.records[-1].token_end, 27)
        controller.seal_tail(
            "request-a", slot_id=7, token_index=28, reason="server_release_guard",
        )
        boundary = controller.boundary(
            "request-a", slot_id=7, token_index=28, at_us=at_us + 1_000, terminal=True,
        ).boundary
        observation = AdaptiveDecodeRawWindowObservation(
            fleet_energy_uj_by_domain={"fleet": 100}, phone_compute_us=0,
            usb_transfer_us=0, rpc_us=0, exposed_tail_us=0, output_valid=True,
            evidence_ids=("physical:terminal-release-confirmed",),
            energy_boundary_id="synthetic-fleet", energy_attribution_kind="diagnostic",
            failure_reason="released_slot_phone_tail",
        )
        for changed, observed, token, when in (
            (boundary, observation, 29, at_us + 3_000),
            (boundary, observation, 30, at_us),
            (boundary, replace(observation, evidence_ids=("synthetic:missing",)),
             30, at_us + 3_000),
            (replace(boundary, applied_ack=AdaptiveDecodePolicyAck(
                request_id="request-a", slot_id=7, plan_generation=100,
                applied_token_index=boundary.token_start, applied_at_us=boundary.started_at_us,
                policy_hash=boundary.policy.policy_hash,
            )), observation, 30, at_us + 3_000),
        ):
            with self.assertRaisesRegex(AdaptiveDecodeError, "transaction differs"):
                controller.discard_stale_window(
                    "request-a", changed, observed, "released_slot_phone_tail",
                    terminal_token_index=token, terminal_at_us=when,
                )
        result = controller.discard_stale_window(
            "request-a", boundary, observation, "released_slot_phone_tail",
            terminal_token_index=30, terminal_at_us=at_us + 3_000,
        )
        self.assertEqual(result.reason, "TAIL_SEALED")
        preview = controller.preview_completion("request-a")
        self.assertEqual(preview.windows, tuple(before.records))
        self.assertEqual(preview.final_policy, before.current_policy)
        self.assertEqual(preview.windows[-1].applied_ack, before.current_ack)
        self.assertEqual(preview.unmeasured_tail_tokens, 3)
        self.assertEqual(preview.unmeasured_tail_reason, "server_release_guard")
        self.assertEqual(AdaptiveDecodeGroupedObservation.from_json(preview.to_json()), preview)
        self.assertEqual(controller.complete("request-a", "COMPLETED"), preview)

    def test_released_baseline_window_requires_exact_terminal_progress(self):
        controller = AdaptiveDecodeController()
        self.start(controller)
        boundary = controller.boundary(
            "request-a", slot_id=7, token_index=3, at_us=3_000,
        ).boundary
        observation = AdaptiveDecodeRawWindowObservation(
            fleet_energy_uj_by_domain={"fleet": 100},
            phone_compute_us=0, usb_transfer_us=0, rpc_us=0, exposed_tail_us=0,
            output_valid=True,
            evidence_ids=("physical:terminal-release-confirmed",),
            energy_boundary_id="synthetic-fleet",
            energy_attribution_kind="diagnostic",
            failure_reason="released_slot_baseline_tail",
            completed_phone_calls=0, completed_phone_input_rows=0,
        )
        for token, at_us in ((29, 30_000), (30, 2_000), (31, 30_000)):
            with self.subTest(token=token, at_us=at_us):
                with self.assertRaisesRegex(AdaptiveDecodeError, "transaction differs"):
                    controller.discard_stale_window(
                        "request-a", boundary, observation,
                        "released_slot_baseline_tail",
                        terminal_token_index=token, terminal_at_us=at_us,
                    )
        result = controller.discard_stale_window(
            "request-a", boundary, observation, "released_slot_baseline_tail",
            terminal_token_index=30, terminal_at_us=30_000,
        )
        self.assertEqual(result.reason, "STALE_SLOT_DIRECTIVE_DISCARDED")
        preview = controller.preview_completion("request-a")
        self.assertEqual(preview.windows[-1].token_end, 30)
        self.assertEqual(preview.windows[-1].finished_at_us, 30_000)
        self.assertFalse(preview.windows[-1].measurement_eligible)
        self.assertEqual(preview.windows[-1].failure_reason, "released_slot_baseline_tail")
        self.assertEqual(preview.unmeasured_tail_tokens, 0)
        self.assertEqual(controller.complete("request-a", "COMPLETED"), preview)

    def test_released_baseline_preserves_historical_zero_ack(self):
        controller = AdaptiveDecodeController()
        directive = self.start(controller)
        directive = self.record(controller, directive.target_token_index, 3_000, 100,
                                completed_phone_calls=0)
        opened = self.acknowledge(controller, directive.control, 3, 3_001)
        controller.helper_unavailable("request-a")
        directive = self.record(controller, opened.target_token_index, 5_000, 60,
                                completed_phone_calls=2)
        self.assertTrue(directive.control.policy.baseline)
        opened = self.acknowledge(controller, directive.control, 5, 5_001)
        continued = self.record(controller, opened.target_token_index, 7_000, 100,
                                completed_phone_calls=0)
        self.assertIsNone(continued.control)
        prior = controller.checkpoint()[1]["request-a"].records
        self.assertEqual(prior[-1].applied_ack.applied_token_index, 5)
        boundary = controller.boundary(
            "request-a", slot_id=7, token_index=continued.target_token_index,
            at_us=9_000,
        ).boundary
        self.assertIsNone(boundary.applied_ack)
        observation = AdaptiveDecodeRawWindowObservation(
            fleet_energy_uj_by_domain={"fleet": 100},
            phone_compute_us=0, usb_transfer_us=0, rpc_us=0, exposed_tail_us=0,
            output_valid=True, evidence_ids=("physical:terminal-release-confirmed",),
            energy_boundary_id="synthetic-fleet", energy_attribution_kind="diagnostic",
            failure_reason="released_slot_baseline_tail",
            completed_phone_calls=0, completed_phone_input_rows=0,
        )
        controller.discard_stale_window(
            "request-a", boundary, observation, "released_slot_baseline_tail",
            terminal_token_index=30, terminal_at_us=30_000,
        )
        result = controller.preview_completion("request-a")
        self.assertEqual(result.windows[:-1], tuple(prior))
        self.assertIsNone(result.windows[-1].applied_ack)
        self.assertFalse(result.windows[-1].measurement_eligible)
        self.assertEqual(result.windows[-1].token_end, 30)
        self.assertEqual(result.windows[-2].applied_ack.applied_token_index, 5)
        self.assertEqual(sum(row.completed_phone_calls for row in result.windows), 2)
        self.assertEqual(controller.complete("request-a", "COMPLETED"), result)

    def test_cohort_membership_change_recovers_phone_policy_to_baseline(self):
        controller = AdaptiveDecodeController()
        directive = self.start(controller)
        directive = self.record(
            controller, directive.target_token_index, 3_000, 100
        )
        opened = self.acknowledge(
            controller, directive.control, 3, 3_001
        )
        controller.seal_tail(
            "request-a",
            slot_id=7,
            token_index=opened.target_token_index,
            reason="cohort_membership_changed",
        )
        directive = self.record(
            controller,
            opened.target_token_index,
            5_001,
            40,
        )

        self.assertIsNotNone(directive.control)
        self.assertTrue(directive.control.policy.baseline)
        self.assertEqual(
            controller.snapshot("request-a")["state"], "RECOVERING"
        )
        sealed = self.acknowledge(
            controller, directive.control, opened.target_token_index, 5_002
        )
        self.assertEqual(sealed.reason, "TAIL_SEALED")
        snapshot = controller.snapshot("request-a")
        self.assertEqual(snapshot["state"], "EXPLOITING")
        self.assertEqual(
            snapshot["current_policy_hash"], self.baseline.policy_hash
        )

    def test_warmup_window_is_excluded_before_first_coarse_probe(self):
        controller = AdaptiveDecodeController()
        config = replace(self.config, warmup_windows_per_policy=1)
        directive = controller.start(
            request_id="request-a",
            ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            baseline=self.baseline,
            candidates=(self.small, self.large),
            output_tokens=30,
            context_length=64,
            active_batch=1,
            deadline_us=1_000_000,
            slot_id=7,
            first_token_index=1,
            first_token_at_us=1_000,
            config=config,
        )

        directive = self.record(
            controller, directive.target_token_index, 3_000, 100
        )
        self.assertIsNone(directive.control)
        self.assertEqual(
            controller.snapshot("request-a")[
                "measurement_eligible_record_count"
            ],
            0,
        )
        directive = self.record(
            controller, directive.target_token_index, 5_000, 100
        )
        self.assertEqual(
            directive.control.policy.split_fraction_ppm, 500_000
        )
        self.assertEqual(
            controller.snapshot("request-a")[
                "measurement_eligible_record_count"
            ],
            1,
        )

    def test_qualified_coarse_leader_exploits_without_refinement(self):
        controller = AdaptiveDecodeController()

        def fraction_policy(fraction: int) -> AdaptiveDecodePolicy:
            return AdaptiveDecodePolicy(
                route_id="phone-" + str(fraction),
                executor_id="desktop-phone",
                operator_plan_sha256=PLAN,
                desktop_parent_route_id="desktop-control",
                desktop_placement_sha256=PLACEMENT,
                layer_indices=(2, 3),
                layer_mask=(1 << 2) | (1 << 3),
                columns=fraction // 100,
                split_fraction_ppm=fraction,
                resource_ids=("cpu", "gpu", "phone", "usb"),
                predicted_latency_per_token_us=1_000,
                predicted_energy_per_token_uj=60,
            )

        candidates = tuple(
            fraction_policy(fraction)
            for fraction in (
                62_500,
                125_000,
                187_500,
                250_000,
                375_000,
                437_500,
                500_000,
                562_500,
                625_000,
                750_000,
                875_000,
                1_000_000,
            )
        )
        config = AdaptiveDecodeConfig(
            minimum_remaining_tokens=4,
            minimum_window_tokens=2,
            maximum_window_tokens=2,
            maximum_probe_tokens=160,
            maximum_probe_candidates=9,
            measurement_resolution_us=1,
            transition_cost_us=1,
            transition_energy_uj=1,
            minimum_energy_saving_ppm=10_000,
            uncertainty_ppm=10_000,
            coarse_probe_fractions_ppm=(
                500_000,
                1_000_000,
                250_000,
                750_000,
            ),
        )
        directive = controller.start(
            request_id="request-a",
            ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            baseline=self.baseline,
            candidates=candidates,
            output_tokens=160,
            context_length=64,
            active_batch=1,
            deadline_us=2_000_000,
            slot_id=7,
            first_token_index=1,
            first_token_at_us=1_000,
            config=config,
        )
        at_us = 1_000
        for _ in range(100):
            snapshot = controller.snapshot("request-a")
            target = snapshot["target_token"]
            self.assertIsNotNone(target)
            at_us += 2_000
            current = snapshot["current_policy_hash"]
            current_policy = next(
                row for row in (self.baseline, *candidates)
                if row.policy_hash == current
            )
            fraction = current_policy.split_fraction_ppm
            energy = (
                100
                if current_policy.baseline
                else 40 + abs(fraction - 500_000) // 25_000
            )
            directive = self.record(
                controller, target, at_us, energy
            )
            if directive.control is not None:
                at_us += 1
                directive = self.acknowledge(
                    controller, directive.control, target, at_us
                )
            if controller.snapshot("request-a")["state"] == "EXPLOITING":
                break
        else:
            self.fail("adaptive coarse-to-fine search did not converge")

        fractions = controller.snapshot("request-a")[
            "probe_fractions_ppm"
        ]
        self.assertEqual(
            fractions[:4],
            (500_000, 1_000_000, 250_000, 750_000),
        )
        self.assertEqual(len(fractions), 4)
        self.assertEqual(controller.active_policy("request-a").split_fraction_ppm, 500_000)
        snapshot = controller.snapshot("request-a")
        self.assertLess(
            snapshot["measurement_eligible_record_count"],
            snapshot["record_count"],
        )

    def test_learning_window_authorizes_each_pending_unmeasured_fraction(self):
        controller = AdaptiveDecodeController()
        higher = policy("phone-higher", 750, (2, 3))
        directive = controller.start(
            request_id="request-a",
            ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            baseline=self.baseline,
            candidates=(self.large, higher),
            output_tokens=30,
            context_length=64,
            active_batch=1,
            deadline_us=1_000_000,
            slot_id=7,
            first_token_index=1,
            first_token_at_us=1_000,
            config=self.config,
            helper_evidence_state="LEARNING",
        )
        scope = SimpleNamespace(_adaptive_decode=controller)

        def may_explore(fraction):
            return HelperPreparationMixin._bounded_learning_exploration_bid(
                scope,
                "request-a",
                controller.helper_window_bid(
                    "request-a", requested_fraction_ppm=fraction
                ),
            )

        directive = self.record(controller, directive.target_token_index, 3_000, 100)
        self.assertEqual(directive.control.policy, higher)
        self.assertTrue(may_explore(750_000))
        self.assertFalse(may_explore(500_000))
        directive = self.acknowledge(controller, directive.control, 3, 3_001)
        directive = self.record(
            controller, directive.target_token_index, 5_001, 70,
            completed_phone_calls=2,
        )
        self.assertEqual(directive.control.policy, self.large)
        self.assertTrue(may_explore(500_000))
        self.assertFalse(may_explore(750_000))

    def test_learning_admission_survives_ack_and_warmup(self):
        controller = AdaptiveDecodeController()
        pessimistic = replace(self.large, predicted_energy_per_token_uj=200)
        opened = controller.start(
            request_id="request-a", ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT, planning_profile_sha256=PLAN,
            baseline=self.baseline, candidates=(pessimistic,), output_tokens=100,
            context_length=64, active_batch=1, deadline_us=1_000_000,
            slot_id=7, first_token_index=1, first_token_at_us=1_000,
            config=replace(self.config, warmup_windows_per_policy=1),
            helper_evidence_state="LEARNING",
        )
        opened = self.record(controller, opened.target_token_index, 3_000, 100)
        pending = self.record(controller, opened.target_token_index, 5_000, 100)
        self.assertTrue(controller.helper_window_bid("request-a")["learning_exploration_eligible"])
        opened = self.acknowledge(controller, pending.control, 5, 5_001)
        self.assertTrue(controller.helper_window_bid("request-a")["learning_exploration_eligible"])
        opened = self.record(controller, opened.target_token_index, 7_001, 160, completed_phone_calls=2)
        self.assertTrue(controller.helper_window_bid("request-a")["learning_exploration_eligible"])
        pending = self.record(controller, opened.target_token_index, 9_001, 160, completed_phone_calls=2)
        self.assertTrue(pending.control.policy.baseline)
        self.assertFalse(controller.helper_window_bid("request-a")["learning_exploration_eligible"])
        self.assertIn(pessimistic.policy_hash, controller.snapshot("request-a")["eliminated_policy_reasons"])

    def test_membership_change_keeps_request_cap_and_discards_mixed_window(self):
        controller = AdaptiveDecodeController()
        opened = self.start(controller)
        pending = self.record(controller, opened.target_token_index, 3_000, 100)
        opened = self.acknowledge(controller, pending.control, 3, 3_001)
        pending = self.record(controller, opened.target_token_index, 5_001, 40,
                              active_batch=1, next_active_batch=2, membership_changed=True)
        session = controller.checkpoint()[1]["request-a"]
        self.assertEqual(session.probe_tokens, 2)
        old_receipt = session.records[-1]
        self.assertFalse(old_receipt.measurement_eligible)
        self.assertEqual(old_receipt.active_batch, 1)
        self.assertEqual(old_receipt.next_active_batch, 2)
        self.assertEqual(AdaptiveDecodeWindowReceipt.from_json(old_receipt.to_json()), old_receipt)
        self.assertFalse(controller._current_valid_records(session, self.large, operational=True))
        self.assertTrue(pending.control.policy.baseline)
        opened = self.acknowledge(controller, pending.control, 5, 5_002)
        # A different membership with the same count is also a new context.
        opened = self.record(controller, opened.target_token_index, 7_002, 100,
                             active_batch=2, next_active_batch=2, membership_changed=True)
        self.assertIsNone(opened.control)
        pending = self.record(controller, opened.target_token_index, 9_002, 100, active_batch=2)
        self.assertIsNotNone(pending.control)
        self.assertFalse(pending.control.policy.baseline)
        session = controller.checkpoint()[1]["request-a"]
        self.assertEqual(session.probe_tokens, 2)
        self.assertEqual(session.records[-3], old_receipt)

    def test_cached_learning_reservation_survives_ack(self):
        controller, directive = self._start_diagnostic_winner(self._diagnostic_winner_history())
        self.assertEqual(directive.reason, "VERIFICATION_RESERVED")
        control = directive.control
        self.assertTrue(controller.helper_window_bid("request-b")["learning_exploration_eligible"])
        controller.acknowledge("request-b", AdaptiveDecodePolicyAck(
            request_id="request-b", slot_id=8, plan_generation=control.plan_generation,
            applied_token_index=1, applied_at_us=1_001, policy_hash=control.policy.policy_hash,
        ))
        self.assertTrue(controller.helper_window_bid("request-b")["learning_exploration_eligible"])

    def test_return_to_singleton_does_not_reuse_pre_context_control_delay(self):
        controller = AdaptiveDecodeController()
        baseline = replace(self.baseline, predicted_latency_per_token_us=628_753)
        candidate = replace(self.large, predicted_latency_per_token_us=500_000)
        config = replace(self.config, minimum_window_tokens=4, maximum_window_tokens=4,
                         maximum_probe_tokens=80, warmup_windows_per_policy=1,
                         maximum_latency_ppm=1_250_000, uncertainty_ppm=100_000)
        opened = controller.start(
            request_id="request-a", ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT, planning_profile_sha256=PLAN,
            baseline=baseline, candidates=(candidate,), output_tokens=230,
            context_length=310, active_batch=1, deadline_us=1,
            slot_id=7, first_token_index=1, first_token_at_us=1_000,
            config=config, helper_evidence_state="LEARNING",
        )
        at_us = 1_000
        for _ in range(2):
            at_us += 4 * 628_753
            opened = self.record(controller, opened.target_token_index, at_us, 100)
        control = opened.control
        self.assertIsNotNone(control)
        at_us += 9_434_196
        opened = controller.acknowledge("request-a", AdaptiveDecodePolicyAck(
            request_id="request-a", slot_id=7, plan_generation=control.plan_generation,
            applied_token_index=10, applied_at_us=at_us, policy_hash=control.policy.policy_hash,
        ), transition_observation=AdaptiveDecodeRawWindowObservation(
            fleet_energy_uj_by_domain={"fleet": 1_000}, phone_compute_us=0,
            usb_transfer_us=0, rpc_us=0, exposed_tail_us=0, output_valid=True,
            evidence_ids=("synthetic:delayed-control",), energy_boundary_id="synthetic-fleet",
            energy_attribution_kind="isolated",
        ))
        self.assertEqual(controller._verification_control_cost(
            controller.checkpoint()[1]["request-a"])[0], 9_434_196)
        at_us += 2_000_000
        opened = self.record(controller, opened.target_token_index, at_us, 60,
                             active_batch=1, next_active_batch=2, membership_changed=True,
                             completed_phone_calls=4)
        self.assertTrue(opened.control.policy.baseline)
        opened = self.acknowledge(controller, opened.control, 14, at_us + 1)
        at_us += 2_500_001
        opened = self.record(controller, opened.target_token_index, at_us, 100,
                             active_batch=2, next_active_batch=1, membership_changed=True)
        session = controller.checkpoint()[1]["request-a"]
        self.assertEqual(session.probe_tokens, 4)
        self.assertEqual(session.state, "PROBING")
        self.assertEqual(controller._verification_control_cost(session)[0], config.transition_cost_us)
        self.assertEqual(len(session.records), session.context_record_start)
        for _ in range(2):
            at_us += 4 * 628_753
            opened = self.record(controller, opened.target_token_index, at_us, 100)
        self.assertIsNotNone(opened.control)
        self.assertEqual(opened.control.policy, candidate)
        token = controller.checkpoint()[1]["request-a"].transition_start_token
        opened = self.acknowledge(controller, opened.control, token, at_us + 1)
        at_us += 1
        for _ in range(2):
            at_us += 2_000_000
            opened = self.record(controller, opened.target_token_index, at_us, 60,
                                 completed_phone_calls=4)
        session = controller.checkpoint()[1]["request-a"]
        self.assertTrue(controller._current_valid_records(session, candidate))
        self.assertEqual(session.probe_tokens, 12)
        self.assertLessEqual(session.probe_tokens, config.maximum_probe_tokens)

    def test_membership_flaps_cannot_replenish_probe_tokens(self):
        controller = AdaptiveDecodeController()
        opened = controller.start(
            request_id="request-a", ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT, planning_profile_sha256=PLAN,
            baseline=self.baseline, candidates=(self.large,), output_tokens=100,
            context_length=64, active_batch=2, deadline_us=1_000_000,
            slot_id=7, first_token_index=1, first_token_at_us=1_000,
            config=replace(self.config, maximum_probe_tokens=7),
            helper_evidence_state="LEARNING",
        )
        token, at_us, count, spent = 1, 1_000, 2, []
        for _ in range(30):
            if opened.control is not None:
                at_us += 1
                opened = self.acknowledge(controller, opened.control, token, at_us)
            active = controller.active_policy("request-a")
            token = opened.target_token_index
            at_us += 2_000
            changed = not active.baseline
            if changed:
                count = 1 if count == 2 else 2
            opened = self.record(controller, token, at_us, 100 if active.baseline else 60,
                                 active_batch=controller.snapshot("request-a")["active_batch"],
                                 next_active_batch=count if changed else None,
                                 membership_changed=changed, completed_phone_calls=0 if active.baseline else 2)
            spent.append(controller.snapshot("request-a")["probe_tokens"])
        self.assertEqual(spent, sorted(spent))
        self.assertGreater(spent[-1], 0)
        self.assertLessEqual(spent[-1], 7)
        self.assertIsNone(opened.control)
        self.assertTrue(controller.active_policy("request-a").baseline)

    def test_insufficient_opportunity_uses_same_attachment_and_probe_budget(self):
        controller = AdaptiveDecodeController()
        controller.start(
            request_id="request-a", ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT, planning_profile_sha256=PLAN,
            baseline=self.baseline, candidates=(self.large,), output_tokens=24,
            context_length=64, active_batch=1, deadline_us=500,
            slot_id=7, first_token_index=1, first_token_at_us=1_000,
            config=replace(self.config, minimum_window_tokens=4, maximum_window_tokens=4,
                           minimum_remaining_tokens=24, warmup_windows_per_policy=1),
            helper_evidence_state="LEARNING",
        )
        self.assertEqual(controller.helper_attachment_opportunity(
            "request-a", token_index=1, at_us=1_000), "INSUFFICIENT_OPPORTUNITY")
        session = controller.checkpoint()[1]["request-a"]
        self.assertFalse(controller._can_probe(session, 1, 1_000))
        self.assertFalse(session.eliminated_policy_reasons)
        self.assertTrue(controller.active_policy("request-a").baseline)

    def test_initial_budget_deferral_retries_after_current_baseline(self):
        controller = AdaptiveDecodeController()
        baseline = replace(self.baseline, predicted_latency_per_token_us=100)
        directive = controller.start(
            request_id="request-a", ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT, planning_profile_sha256=PLAN,
            baseline=baseline, candidates=(self.large,), output_tokens=80,
            context_length=64, active_batch=1, deadline_us=2_000,
            slot_id=7, first_token_index=1, first_token_at_us=1_000,
            config=self.config, helper_evidence_state="LEARNING",
        )
        self.assertEqual(directive.state, "EXPLOITING")
        self.assertEqual(controller.helper_attachment_opportunity(
            "request-a", token_index=1, at_us=1_000), "INSUFFICIENT_OPPORTUNITY")
        directive = self.record(controller, directive.target_token_index, 3_000, 100)
        session = controller.checkpoint()[1]["request-a"]
        self.assertTrue(controller._can_probe(session, 3, 3_000))
        self.assertIsNotNone(directive.control)
        self.assertEqual(directive.control.policy, self.large)
        self.assertEqual(session.probe_tokens, 0)
        self.assertEqual(directive.state, "PROBING")

    def test_historical_baseline_start_completes_current_pair_before_exploitation(self):
        full = policy("phone-full", 1000, (2, 3))
        seventy_five = policy("phone-seventy-five", 750, (2, 3))
        controller, directive = self._start_diagnostic_winner(
            self._diagnostic_winner_history(), candidates=(full, seventy_five),
            output_tokens=200,
        )
        self.assertEqual(directive.control.policy, full)
        self.assertFalse(controller._current_valid_records(
            controller.checkpoint()[1]["request-b"], self.baseline, operational=True))
        self._verify_diagnostic_winner(controller, directive, 40)
        session = controller.checkpoint()[1]["request-b"]
        self.assertTrue(controller._current_valid_records(
            session, self.baseline, operational=True))
        self.assertFalse(controller.active_policy("request-b").baseline)
        self.assertLessEqual(session.probe_tokens, session.config.maximum_probe_tokens)
        self.assertTrue(controller._qualifies(
            session, controller.active_policy("request-b"),
            session.window_start_token, session.window_start_us))

    def test_dominated_warmup_keeps_qualified_measured_winner(self):
        controller = AdaptiveDecodeController()
        full = policy("phone-full", 1000, (2, 3))
        directive = controller.start(
            request_id="request-a", ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT, planning_profile_sha256=PLAN,
            baseline=self.baseline, candidates=(full, self.small), output_tokens=200,
            context_length=64, active_batch=1, deadline_us=1_000_000,
            slot_id=7, first_token_index=1, first_token_at_us=1_000,
            config=replace(self.config, warmup_windows_per_policy=1,
                           maximum_probe_tokens=80), helper_evidence_state="LEARNING",
        )
        at_us = 1_000
        token = directive.target_token_index
        for energy, duration in ((100, 2_000), (100, 2_000), (40, 1_000), (40, 1_000)):
            if directive.control is not None:
                at_us += 1
                directive = self.acknowledge(controller, directive.control, token, at_us)
            token = directive.target_token_index
            at_us += duration
            directive = self.record(controller, token, at_us, energy,
                                    completed_phone_calls=0 if energy == 100 else 2)
        self.assertEqual(directive.control.policy, self.small)
        checkpoint = controller.checkpoint()
        session = checkpoint[1]["request-a"]
        measured = session.records[-1]
        identity = controller._policy_identity(self.small)
        session.historical_records[identity] = (replace(
            measured, policy=self.small,
            fleet_energy_uj_by_domain={"fleet": 80 * measured.token_count},
        ),)
        session.historical_group_counts[identity] = 1
        controller.restore(checkpoint)
        at_us += 1
        directive = self.acknowledge(controller, directive.control, token, at_us)
        token = directive.target_token_index
        at_us += 1_000
        directive = self.record(controller, token, at_us, 80, completed_phone_calls=2)
        session = controller.checkpoint()[1]["request-a"]
        self.assertEqual(session.eliminated_policy_reasons[self.small.policy_hash],
                         "ENERGY_DOMINATED")
        self.assertTrue(controller._qualifies(session, full, token, at_us))
        self.assertEqual(directive.reason, "PROBE_CANDIDATE_REJECTED")
        self.assertEqual(directive.control.policy, full)
        self.assertEqual(directive.state, "EXPLOITING")
        self.assertLessEqual(session.probe_tokens, session.config.maximum_probe_tokens)

    def test_learning_rebind_keeps_service_and_verifies_only_the_prior_fraction(self):
        controller = AdaptiveDecodeController()
        higher = policy("phone-higher", 750, (2, 3))
        directive = controller.start(
            request_id="request-a",
            ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            baseline=self.baseline,
            candidates=(self.large, higher),
            output_tokens=40,
            context_length=64,
            active_batch=1,
            deadline_us=1_000_000,
            slot_id=7,
            first_token_index=1,
            first_token_at_us=1_000,
            config=self.config,
            helper_evidence_state="LEARNING",
        )
        directive = self.record(controller, directive.target_token_index, 3_000, 100)
        directive = self.acknowledge(controller, directive.control, 3, 3_001)
        expanded = tuple(
            replace(row, layer_indices=(2, 3, 4), layer_mask=28)
            for row in (self.large, higher)
        )
        arguments = dict(
            phone_layout_generation=2,
            phone_layout_geometry_sha256="sha256:" + "4" * 64,
            candidates=expanded,
            component_capability_sha256=PLAN,
            ticket_policy=None,
            helper_evidence_state="LEARNING",
        )
        controller.helper_rebound("request-a", **arguments)
        self.assertEqual(controller.active_policy("request-a"), higher)
        self.assertEqual(controller.snapshot("request-a")["state"], "PROBING")
        before_retry = controller.snapshot("request-a")
        controller.helper_rebound("request-a", **arguments)
        self.assertEqual(controller.snapshot("request-a"), before_retry)
        directive = self.record(
            controller, directive.target_token_index, 5_001, 70,
            completed_phone_calls=2,
        )
        self.assertEqual(directive.control.policy, expanded[1])
        scope = SimpleNamespace(_adaptive_decode=controller)
        self.assertTrue(HelperPreparationMixin._bounded_learning_exploration_bid(
            scope, "request-a", controller.helper_window_bid(
                "request-a", requested_fraction_ppm=750_000
            )
        ))
        # Only the running fraction's expanded form is probed: no 500 ppm
        # candidate is appended after the contract change.
        self.assertEqual(controller.snapshot("request-a")["probe_fractions_ppm"], (750_000,))
        directive = self.acknowledge(controller, directive.control, 5, 5_002)
        directive = self.record(
            controller, directive.target_token_index, 7_002, 60,
            completed_phone_calls=2,
        )
        # The prior is verified against the same baseline and exploited; the
        # sweep does not resume and the budget is only spent, never refilled.
        self.assertIsNone(directive.control)
        self.assertEqual(directive.state, "EXPLOITING")
        session = controller.checkpoint()[1]["request-a"]
        self.assertEqual(session.incumbent_policy, expanded[1])
        self.assertEqual(controller.snapshot("request-a")["probe_tokens"], 4)
        self.assertEqual(session.verification_attempts, 0)

    def test_learning_rebind_adds_only_the_measured_leader_as_second_prior(self):
        controller = AdaptiveDecodeController()
        quarter = policy("phone-quarter", 250, (2, 3))
        half = policy("phone-half", 500, (2, 3))
        three_quarter = policy("phone-three-quarter", 750, (2, 3))
        full = policy("phone-full", 1000, (2, 3))
        directive = controller.start(
            request_id="request-a", ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT, planning_profile_sha256=PLAN,
            baseline=self.baseline, candidates=(quarter, half, three_quarter, full),
            output_tokens=60, context_length=64, active_batch=1, deadline_us=1_000_000,
            slot_id=7, first_token_index=1, first_token_at_us=1_000,
            config=replace(self.config, maximum_probe_candidates=4, maximum_probe_tokens=40),
            helper_evidence_state="LEARNING",
        )
        at_us = 1_000
        energies = {full.policy_hash: 60, three_quarter.policy_hash: 80}
        # Coarse sweep order: 100%, 75%, then 50% is running when the layout expands.
        at_us += 2_000
        directive = self.record(controller, directive.target_token_index, at_us, 100)
        self.assertEqual(directive.control.policy, full)
        for expected_next in (three_quarter, half):
            directive = self.acknowledge(controller, directive.control, directive.target_token_index
                                         or controller.checkpoint()[1]["request-a"].transition_start_token, at_us)
            at_us += 2_000
            current = controller.checkpoint()[1]["request-a"].current_policy
            directive = self.record(controller, directive.target_token_index, at_us,
                                    energies[current.policy_hash], completed_phone_calls=2)
            self.assertEqual(directive.control.policy, expected_next)
        directive = self.acknowledge(controller, directive.control, directive.target_token_index
                                     or controller.checkpoint()[1]["request-a"].transition_start_token, at_us)
        self.assertEqual(controller.active_policy("request-a"), half)
        probe_tokens_before = controller.snapshot("request-a")["probe_tokens"]
        expanded = {row.policy_hash: replace(row, layer_indices=(2, 3, 4), layer_mask=28)
                    for row in (quarter, half, three_quarter, full)}
        controller.helper_rebound(
            "request-a", phone_layout_generation=2,
            phone_layout_geometry_sha256="sha256:" + "4" * 64,
            candidates=tuple(expanded.values()), component_capability_sha256=PLAN,
            ticket_policy=None, helper_evidence_state="LEARNING",
        )
        snapshot = controller.snapshot("request-a")
        # Continuation of the running 50% first, then the measured leader (100%
        # at 60 beats 75% at 80); the unmeasured 25% and the losing 75% are not
        # re-probed under the new contract.
        self.assertEqual(snapshot["probe_fractions_ppm"], (500_000, 1_000_000))
        self.assertEqual(snapshot["probe_tokens"], probe_tokens_before)
        session = controller.checkpoint()[1]["request-a"]
        self.assertEqual(session.pending_helper_refresh_policy, expanded[half.policy_hash])
        # Measure the continuation, then the leader; the leader wins and is exploited.
        at_us += 2_000
        directive = self.record(controller, directive.target_token_index, at_us, 90, completed_phone_calls=2)
        self.assertEqual(directive.control.policy, expanded[half.policy_hash])
        directive = self.acknowledge(controller, directive.control,
                                     controller.checkpoint()[1]["request-a"].transition_start_token, at_us)
        at_us += 2_000
        directive = self.record(controller, directive.target_token_index, at_us, 90, completed_phone_calls=2)
        self.assertEqual(directive.control.policy, expanded[full.policy_hash])
        directive = self.acknowledge(controller, directive.control,
                                     controller.checkpoint()[1]["request-a"].transition_start_token, at_us)
        at_us += 2_000
        directive = self.record(controller, directive.target_token_index, at_us, 55, completed_phone_calls=2)
        self.assertIsNone(directive.control)
        self.assertEqual(directive.state, "EXPLOITING")
        session = controller.checkpoint()[1]["request-a"]
        self.assertEqual(session.incumbent_policy, expanded[full.policy_hash])
        self.assertEqual(controller.active_policy("request-a"), expanded[full.policy_hash])
        controlled = {row.policy.split_fraction_ppm for row in session.records if not row.policy.baseline}
        self.assertNotIn(250_000, controlled)

    def test_default_probe_selects_and_exploits_without_baseline_between(self):
        controller = AdaptiveDecodeController()
        seventy_five = policy("phone-seventy-five", 750, (2, 3))
        directive = controller.start(
            request_id="request-a",
            ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            baseline=self.baseline,
            candidates=(self.small, self.large, seventy_five),
            output_tokens=30,
            context_length=64,
            active_batch=1,
            deadline_us=1_000_000,
            slot_id=7,
            first_token_index=1,
            first_token_at_us=1_000,
            config=self.config,
        )

        directive = self.record(
            controller, directive.target_token_index, 3_000, 100
        )
        self.assertEqual(
            directive.control.policy.split_fraction_ppm, 750_000
        )
        directive = self.acknowledge(
            controller, directive.control, 3, 3_001
        )
        directive = self.record(
            controller, directive.target_token_index, 5_001, 70
        )
        self.assertEqual(
            directive.control.policy.split_fraction_ppm, 500_000
        )
        directive = self.acknowledge(
            controller, directive.control, 5, 5_002
        )
        directive = self.record(
            controller, directive.target_token_index, 7_002, 40
        )

        self.assertIsNone(directive.control)
        snapshot = controller.snapshot("request-a")
        self.assertEqual(snapshot["state"], "EXPLOITING")
        self.assertEqual(
            snapshot["current_policy_hash"], self.large.policy_hash
        )
        self.assertEqual(snapshot["probe_fractions_ppm"], (750_000, 500_000))
        session = controller.checkpoint()[1]["request-a"]
        self.assertEqual(
            tuple(row.policy.split_fraction_ppm for row in session.records),
            (0, 750_000, 500_000),
        )
        self.assertEqual(session.records[-1].window_role, "exploration")
        self.assertEqual(directive.state, "EXPLOITING")

        directive = self.record(
            controller, directive.target_token_index, 9_002, 40
        )
        session = controller.checkpoint()[1]["request-a"]
        self.assertEqual(session.records[-1].window_role, "exploitation")
        self.assertEqual(
            session.records[-1].policy.policy_hash,
            self.large.policy_hash,
        )

    def test_warmup_then_winner_has_no_long_desktop_tail(self):
        controller = AdaptiveDecodeController()
        fifty = policy("phone-fifty", 500, (2, 3))
        seventy_five = policy("phone-seventy-five", 750, (2, 3))
        config = AdaptiveDecodeConfig(
            minimum_remaining_tokens=24,
            minimum_window_tokens=24,
            maximum_window_tokens=24,
            maximum_probe_tokens=120,
            maximum_probe_candidates=2,
            measurement_resolution_us=1,
            transition_cost_us=1,
            transition_energy_uj=1,
            minimum_energy_saving_ppm=10_000,
            uncertainty_ppm=100_000,
            warmup_windows_per_policy=1,
        )
        directive = controller.start(
            request_id="request-a",
            ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            baseline=self.baseline,
            candidates=(fifty, seventy_five),
            output_tokens=341,
            context_length=64,
            active_batch=1,
            deadline_us=10_000_000,
            slot_id=7,
            first_token_index=1,
            first_token_at_us=1_000,
            config=config,
        )
        policies = (self.baseline, fifty, seventy_five)
        at_us = 1_000
        for _ in range(30):
            snapshot = controller.snapshot("request-a")
            target = snapshot["target_token"]
            at_us += 24_000
            current = next(
                row for row in policies
                if row.policy_hash == snapshot["current_policy_hash"]
            )
            directive = self.record(
                controller,
                target,
                at_us,
                100 if current.baseline else
                    70 if current == fifty else 40,
            )
            if directive.control is not None:
                at_us += 1
                directive = self.acknowledge(
                    controller, directive.control, target, at_us
                )
            if controller.snapshot("request-a")["state"] == "EXPLOITING":
                break
        else:
            self.fail("adaptive warmup probe did not select a winner")

        self.assertEqual(
            controller.snapshot("request-a")["current_policy_hash"],
            seventy_five.policy_hash,
        )
        session = controller.checkpoint()[1]["request-a"]
        self.assertEqual(
            tuple(row.policy.split_fraction_ppm for row in session.records),
            (0, 0, 750_000, 750_000, 500_000, 500_000),
        )
        while controller.snapshot("request-a")["target_token"] < 341:
            target = controller.snapshot("request-a")["target_token"]
            at_us += 24_000
            directive = self.record(controller, target, at_us, 40)
            self.assertIsNone(directive.control)
        target = controller.snapshot("request-a")["target_token"]
        at_us += 24_000
        self.record(controller, target, at_us, 40)
        grouped = controller.complete("request-a", "COMPLETED")
        assisted_tokens = sum(
            row.token_count for row in grouped.windows
            if not row.policy.baseline
        )
        self.assertGreaterEqual(assisted_tokens, 239)
        self.assertTrue(all(
            not row.policy.baseline
            for row in grouped.windows[2:]
        ))
        self.assertEqual(grouped.final_policy, seventy_five)

    def test_assumed_phone_power_is_operational_only_when_enabled(self):
        def run(allowed: bool):
            controller = AdaptiveDecodeController()
            config = replace(
                self.config,
                maximum_probe_candidates=1,
                allow_assumed_phone_power_for_operational_selection=allowed,
            )
            directive = controller.start(
                request_id="request-a",
                ticket_id="request-a:attempt:0",
                model_artifact_sha256=ARTIFACT,
                planning_profile_sha256=PLAN,
                baseline=self.baseline,
                candidates=(self.large,),
                output_tokens=30,
                context_length=64,
                active_batch=1,
                deadline_us=1_000_000,
                slot_id=7,
                first_token_index=1,
                first_token_at_us=1_000,
                config=config,
            )
            at_us = 1_000
            for _ in range(20):
                snapshot = controller.snapshot("request-a")
                target = snapshot["target_token"]
                at_us += 2_000
                boundary = controller.boundary(
                    "request-a",
                    slot_id=7,
                    token_index=target,
                    at_us=at_us,
                ).boundary
                directive = controller.record_window(
                    "request-a",
                    boundary,
                    AdaptiveDecodeRawWindowObservation(
                        fleet_energy_uj_by_domain={
                            "fleet": (
                                100 if boundary.policy.baseline else 40
                            ) * boundary.token_count
                        },
                        phone_compute_us=(
                            0 if boundary.policy.baseline else 10
                        ),
                        usb_transfer_us=(
                            0 if boundary.policy.baseline else 5
                        ),
                        rpc_us=0,
                        exposed_tail_us=0,
                        output_valid=True,
                        evidence_ids=(
                            "ASSUMED_4P5W",
                            "physical:nvml-board-power",
                            "physical:rapl-package-0",
                        ),
                        energy_boundary_id="synthetic-fleet",
                        energy_attribution_kind="diagnostic",
                    ),
                )
                if directive.control is not None:
                    at_us += 1
                    directive = controller.acknowledge(
                        "request-a",
                        AdaptiveDecodePolicyAck(
                            request_id="request-a",
                            slot_id=7,
                            plan_generation=(
                                directive.control.plan_generation
                            ),
                            applied_token_index=target,
                            applied_at_us=at_us,
                            policy_hash=directive.control.policy.policy_hash,
                        ),
                    )
                if controller.snapshot("request-a")["state"] == "EXPLOITING":
                    break
            else:
                self.fail("adaptive assumed-power probe did not converge")
            return controller

        disabled = run(False)
        self.assertEqual(
            disabled.snapshot("request-a")["current_policy_hash"],
            self.baseline.policy_hash,
        )
        enabled = run(True)
        self.assertEqual(
            enabled.snapshot("request-a")["current_policy_hash"],
            self.large.policy_hash,
        )
        enabled_session = enabled.checkpoint()[1]["request-a"]
        self.assertFalse(any(
            row.energy_measurement_eligible
            for row in enabled_session.records
        ))
        isolated_assumed = replace(
            enabled_session.records[-1],
            energy_attribution_kind="isolated",
        )
        self.assertFalse(isolated_assumed.energy_measurement_eligible)

    def test_control_acknowledgement_records_in_flight_tokens(self):
        controller = AdaptiveDecodeController()
        directive = self.start(controller)
        directive = self.record(
            controller, directive.target_token_index, 3_000, 100
        )
        control = directive.control
        acknowledgement = AdaptiveDecodePolicyAck(
            request_id="request-a",
            slot_id=7,
            plan_generation=control.plan_generation,
            applied_token_index=5,
            applied_at_us=5_000,
            policy_hash=control.policy.policy_hash,
        )
        checkpoint = controller.checkpoint()
        with self.assertRaisesRegex(
            AdaptiveDecodeError, "transition observation is absent"
        ):
            controller.acknowledge("request-a", acknowledgement)
        self.assertEqual(controller.checkpoint(), checkpoint)

        opened = controller.acknowledge(
            "request-a",
            acknowledgement,
            transition_observation=AdaptiveDecodeRawWindowObservation(
                fleet_energy_uj_by_domain={"fleet": 200},
                phone_compute_us=0,
                usb_transfer_us=0,
                rpc_us=20,
                exposed_tail_us=0,
                output_valid=True,
                evidence_ids=("synthetic:control-transition",),
                energy_boundary_id="synthetic-fleet",
                energy_attribution_kind="isolated",
            ),
        )

        self.assertEqual(controller.snapshot("request-a")["record_count"], 2)
        self.assertEqual(opened.target_token_index, 7)
        self.assertEqual(
            controller.snapshot("request-a")["current_policy_hash"],
            self.large.policy_hash,
        )

    def test_phone_control_failure_returns_to_desktop_once(self):
        controller = AdaptiveDecodeController()
        directive = self.start(controller)
        directive = self.record(
            controller, directive.target_token_index, 3_000, 100
        )
        failed = directive.control
        recovery = controller.control_failed(
            "request-a", failed, "synthetic phone failure", at_us=3_050
        )
        self.assertEqual(recovery.state, "RECOVERING")
        self.assertTrue(recovery.control.policy.baseline)
        opened = self.acknowledge(
            controller, recovery.control, 3, 3_100
        )
        self.assertEqual(opened.state, "BASELINE")
        self.assertEqual(
            controller.snapshot("request-a")["state_history"],
            ("BASELINE", "PREPARING", "PROBING", "RECOVERING", "BASELINE"),
        )

    def test_busy_helper_control_retries_without_abandoning_probe(self):
        controller = AdaptiveDecodeController()
        directive = self.start(controller)
        directive = self.record(
            controller, directive.target_token_index, 3_000, 100
        )
        first_control = directive.control
        deferred = controller.defer_control(
            "request-a",
            first_control,
            "HELPER_RESOURCES_BUSY",
            at_us=3_050,
        )

        self.assertEqual(deferred.reason, "HELPER_CONTROL_DEFERRED")
        self.assertIsNone(deferred.control)
        self.assertEqual(
            controller.snapshot("request-a")["state_history"],
            ("BASELINE", "PREPARING", "PROBING"),
        )
        retried = self.record(
            controller, deferred.target_token_index, 5_050, 100
        )
        self.assertEqual(
            retried.control.policy.policy_hash,
            first_control.policy.policy_hash,
        )
        self.assertGreater(
            retried.control.plan_generation,
            first_control.plan_generation,
        )
        opened = self.acknowledge(
            controller,
            retried.control,
            deferred.target_token_index,
            5_051,
        )
        snapshot = controller.snapshot("request-a")
        self.assertEqual(snapshot["deferred_control_count"], 1)
        self.assertIsNone(snapshot["deferred_policy_hash"])
        self.assertEqual(
            snapshot["current_policy_hash"],
            first_control.policy.policy_hash,
        )
        self.assertIsNotNone(opened.target_token_index)

    def test_deferred_probe_reserves_a_complete_pair_at_actual_retry(self):
        controller = AdaptiveDecodeController()
        opened = controller.start(
            request_id="request-a", ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT, planning_profile_sha256=PLAN,
            baseline=self.baseline, candidates=(self.large,), output_tokens=100,
            context_length=64, active_batch=1, deadline_us=1_000_000,
            slot_id=7, first_token_index=1, first_token_at_us=1_000,
            config=replace(self.config, warmup_windows_per_policy=1),
            helper_evidence_state="LEARNING",
        )
        opened = self.record(controller, opened.target_token_index, 3_000, 100)
        opened = self.record(controller, opened.target_token_index, 5_000, 100)
        first_budget = controller.checkpoint()[1]["request-a"].probe_budget
        at_us = 5_000
        for _ in range(3):
            opened = controller.defer_control(
                "request-a", opened.control, "HELPER_RESOURCES_BUSY", at_us=at_us + 1,
            )
            at_us += 2_001
            opened = self.record(controller, opened.target_token_index, at_us, 100)
        session = controller.checkpoint()[1]["request-a"]
        self.assertEqual(session.probe_tokens, 0)
        self.assertEqual(session.probe_budget["started_at_us"], at_us)
        self.assertGreater(session.probe_budget["token_limit"], first_budget["token_limit"])
        opened = self.acknowledge(controller, opened.control, 11, at_us + 1)
        at_us += 1
        for _ in range(2):
            at_us += 2_000
            opened = self.record(controller, opened.target_token_index, at_us, 60,
                                 completed_phone_calls=2)
        session = controller.checkpoint()[1]["request-a"]
        self.assertNotEqual(opened.reason, "PROBE_INCOMPLETE")
        self.assertTrue(controller._current_valid_records(session, self.large))
        self.assertEqual(session.probe_tokens, 4)

    def test_deferred_eliminated_verification_candidate_advances(self):
        """Physical treatment-4: the verification candidate control was
        deferred (HELPER_WINDOW_NOT_SELECTED) after the candidate had been
        eliminated; the next baseline window dropped the deferred policy and
        then failed with "adaptive verification candidate differs"."""
        controller = AdaptiveDecodeController()
        opened = controller.start(
            request_id="request-a", ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT, planning_profile_sha256=PLAN,
            baseline=self.baseline, candidates=(self.large,), output_tokens=100,
            context_length=64, active_batch=1, deadline_us=1_000_000,
            slot_id=7, first_token_index=1, first_token_at_us=1_000,
            config=self.config, helper_evidence_state="LEARNING",
        )
        opened = self.record(controller, opened.target_token_index, 3_000, 100)
        self.assertIsNotNone(opened.control)
        self.assertFalse(opened.control.policy.baseline)
        # Reach the issued verification_candidate control of the sequencer.
        session = controller._session("request-a")
        session.stage = "verification_candidate"
        session.verification_policy = opened.control.policy
        session.eliminated_policy_reasons[
            opened.control.policy.policy_hash
        ] = "LATENCY_BOUND_EXCEEDED"
        opened = controller.defer_control(
            "request-a", opened.control, "HELPER_WINDOW_NOT_SELECTED",
            at_us=3_001,
        )
        self.assertTrue(controller.active_policy("request-a").baseline)

        opened = self.record(controller, opened.target_token_index, 5_001, 100)

        session = controller._session("request-a")
        self.assertIsNone(session.deferred_policy)
        self.assertIsNone(session.verification_policy)
        self.assertNotEqual(session.stage, "verification_candidate")
        self.assertTrue(
            opened.control is None or opened.control.policy.baseline
        )
        opened = self.record(controller, opened.target_token_index, 7_001, 100)
        self.assertTrue(controller.active_policy("request-a").baseline)

    def test_deferred_probe_cannot_bypass_exhausted_opportunity(self):
        controller = AdaptiveDecodeController()
        opened = controller.start(
            request_id="request-a", ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT, planning_profile_sha256=PLAN,
            baseline=self.baseline, candidates=(self.large,), output_tokens=20,
            context_length=64, active_batch=1, deadline_us=1_000_000,
            slot_id=7, first_token_index=1, first_token_at_us=1_000,
            config=replace(self.config, minimum_remaining_tokens=16),
            helper_evidence_state="LEARNING",
        )
        opened = self.record(controller, opened.target_token_index, 3_000, 100)
        opened = controller.defer_control(
            "request-a", opened.control, "HELPER_RESOURCES_BUSY", at_us=3_001,
        )
        opened = self.record(controller, opened.target_token_index, 5_001, 100)
        self.assertEqual(opened.reason, "INSUFFICIENT_OPPORTUNITY")
        self.assertIsNone(opened.control)
        self.assertIsNone(controller.checkpoint()[1]["request-a"].deferred_policy)
        opened = self.record(controller, opened.target_token_index, 7_001, 100)
        self.assertIsNone(opened.control)
        self.assertTrue(controller.active_policy("request-a").baseline)
        self.assertEqual(controller.snapshot("request-a")["probe_tokens"], 0)

    def test_promising_overlap_collects_evidence_before_exploitation(self):
        controller = AdaptiveDecodeController()
        config = replace(
            self.config,
            uncertainty_ppm=100_000,
            maximum_probe_tokens=40,
        )
        directive = controller.start(
            request_id="request-a",
            ticket_id="request-a:attempt:0",
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            baseline=self.baseline,
            candidates=(self.small,),
            output_tokens=80,
            context_length=64,
            active_batch=1,
            deadline_us=2_000_000,
            slot_id=7,
            first_token_index=1,
            first_token_at_us=1_000,
            config=config,
        )
        at_us = 1_000
        for _ in range(40):
            snapshot = controller.snapshot("request-a")
            target = snapshot["target_token"]
            at_us += 2_000
            directive = self.record(
                controller,
                target,
                at_us,
                100 if snapshot["current_policy_hash"] ==
                    self.baseline.policy_hash else 84,
            )
            if directive.control is not None:
                at_us += 1
                directive = self.acknowledge(
                    controller, directive.control, target, at_us
                )
            if controller.snapshot("request-a")["state"] == "EXPLOITING":
                break
        else:
            self.fail("adaptive confidence extension did not converge")

        session = controller.checkpoint()[1]["request-a"]
        candidate_records = tuple(
            row for row in session.records
            if row.policy == self.small and row.measurement_eligible
        )
        self.assertGreaterEqual(len(candidate_records), 4)
        self.assertEqual(
            controller.snapshot("request-a")["current_policy_hash"],
            self.small.policy_hash,
        )

    def _run(
        self,
        phone_energy: int,
        *,
        deadline_us: int = 1_000_000,
        attribution_kind: str = "isolated",
    ):
        controller = AdaptiveDecodeController()
        directive = self.start(controller, deadline_us=deadline_us)
        at_us = 1_000
        while True:
            snapshot = controller.snapshot("request-a")
            target = snapshot["target_token"]
            if target is None:
                self.fail("adaptive controller has no active token window")
            at_us += 2_000
            directive = self.record(
                controller,
                target,
                at_us,
                100 if snapshot["current_policy_hash"] == self.baseline.policy_hash
                else phone_energy,
                attribution_kind=attribution_kind,
            )
            if directive.control is not None:
                at_us += 1
                directive = self.acknowledge(
                    controller, directive.control, target, at_us
                )
            if controller.snapshot("request-a")["state"] == "EXPLOITING":
                break
        final_policy_hash = controller.snapshot(
            "request-a"
        )["current_policy_hash"]
        while True:
            target = controller.snapshot("request-a")["target_token"]
            at_us += 2_000
            directive = controller.boundary(
                "request-a",
                slot_id=7,
                token_index=target,
                at_us=at_us,
                terminal=target == 30,
            )
            boundary = directive.boundary
            directive = controller.record_window(
                "request-a",
                boundary,
                AdaptiveDecodeRawWindowObservation(
                    fleet_energy_uj_by_domain={
                        "fleet": (
                            100 if boundary.policy.baseline else phone_energy
                        ) * boundary.token_count
                    },
                    phone_compute_us=0,
                    usb_transfer_us=0,
                    rpc_us=0,
                    exposed_tail_us=0,
                    output_valid=True,
                    evidence_ids=("synthetic:window",),
                    energy_boundary_id="synthetic-fleet",
                    energy_attribution_kind=attribution_kind,
                ),
            )
            if target == 30:
                break
            if directive.control is not None:
                at_us += 1
                directive = self.acknowledge(controller, directive.control, target, at_us)
        final_policy_hash = controller.snapshot("request-a")["current_policy_hash"]
        preview = controller.preview_completion(
            "request-a", "COMPLETED"
        )
        self.assertNotEqual(
            controller.snapshot("request-a")["state"], "COMPLETED"
        )
        grouped = controller.complete("request-a", "COMPLETED")
        self.assertEqual(
            grouped.grouped_observation_sha256,
            preview.grouped_observation_sha256,
        )
        return final_policy_hash, grouped

    def test_micro_abba_exploits_conservatively_positive_phone(self):
        final_hash, grouped = self._run(40)
        self.assertNotEqual(final_hash, self.baseline.policy_hash)
        self.assertEqual(grouped.final_policy.policy_hash, final_hash)
        self.assertEqual(grouped.state_history[-1], "COMPLETED")
        self.assertEqual(
            [row.window_index for row in grouped.windows],
            list(range(len(grouped.windows))),
        )
        for left, right in zip(grouped.windows, grouped.windows[1:]):
            self.assertEqual(
                right.previous_record_sha256,
                left.record_sha256.removeprefix("sha256:"),
            )
        assisted = next(
            row for row in grouped.windows
            if not row.policy.baseline and row.usb_transfer_us > 0
        )
        self.assertEqual(assisted.usb_upload_bytes, 20)
        self.assertEqual(assisted.usb_download_bytes, 20)
        self.assertEqual(
            assisted.usb_payload_bandwidth_bytes_per_s, 8_000_000
        )
        self.assertEqual(assisted.desktop_compute_us, 12)
        self.assertEqual(assisted.useful_overlap_us, 8)
        self.assertEqual(assisted.request_queue_delay_us, 3)
        self.assertTrue(any(
            row.window_role == "exploration" for row in grouped.windows
        ))
        self.assertTrue(any(
            row.window_role == "exploitation" for row in grouped.windows
        ))

    def test_energy_negative_phone_keeps_desktop_baseline(self):
        final_hash, grouped = self._run(130)
        self.assertEqual(final_hash, self.baseline.policy_hash)
        self.assertTrue(grouped.final_policy.baseline)

    def test_diagnostic_energy_updates_latency_but_cannot_select_phone(self):
        final_hash, grouped = self._run(
            1, attribution_kind="diagnostic"
        )

        self.assertEqual(final_hash, self.baseline.policy_hash)
        self.assertTrue(any(
            row.measurement_eligible for row in grouped.windows
        ))
        self.assertFalse(any(
            row.energy_measurement_eligible for row in grouped.windows
        ))

    def test_tardy_long_decode_probes_against_baseline_finish_bound(self):
        final_hash, grouped = self._run(40, deadline_us=500)

        self.assertNotEqual(final_hash, self.baseline.policy_hash)
        self.assertIn("PROBING", grouped.state_history)
        self.assertEqual(grouped.final_policy.policy_hash, final_hash)

    def test_group_hash_is_deterministic_and_windows_are_correlated(self):
        _, first = self._run(40)
        _, second = self._run(40)
        self.assertEqual(
            first.grouped_observation_sha256,
            second.grouped_observation_sha256,
        )
        self.assertGreater(len(first.windows), 1)
        self.assertEqual(
            {row.request_id for row in first.windows}, {first.request_id}
        )

        source = AdaptiveDecodeController()
        self.start(source)
        at_us = 1_000
        while True:
            target = source.snapshot("request-a")["target_token"]
            at_us += 2_000
            directive = self.record(source, target, at_us, 40)
            if directive.control is not None:
                at_us += 1
                self.acknowledge(source, directive.control, target, at_us)
            if target == 30:
                break
        source.complete("request-a", "COMPLETED")
        snapshot = source.observation_snapshot()
        restored = AdaptiveDecodeController()
        restored.load_observations(dict(snapshot))
        self.assertEqual(restored.observation_snapshot(), snapshot)
        self.assertEqual(
            restored.observation_state()["grouped_observations"], 1
        )
        self.assertGreater(
            restored.observation_state()["valid_windows"], 1
        )

    def test_later_learning_request_starts_with_persisted_best_fraction(self):
        winner_hash, first = self._run(40)
        self.assertIn(winner_hash, {self.small.policy_hash, self.large.policy_hash})
        previous = "0" * 64
        windows = []
        for row in first.windows:
            acknowledgement = (
                None
                if row.applied_ack is None
                else replace(row.applied_ack, request_id="request-b")
            )
            cloned = replace(
                row,
                request_id="request-b",
                applied_ack=acknowledgement,
                previous_record_sha256=previous,
            )
            windows.append(cloned)
            previous = cloned.record_sha256.removeprefix("sha256:")
        second = replace(
            first,
            request_id="request-b",
            ticket_id="request-b:attempt:0",
            windows=tuple(windows),
        )
        body = {
            "groups": [first.to_json(), second.to_json()],
            "schema": ADAPTIVE_OBSERVATION_STORE_SCHEMA,
        }
        controller = AdaptiveDecodeController()
        controller.load_observations({
            **body,
            "store_sha256": canonical_sha256(body),
        })

        directive = controller.start(
            request_id="request-c",
            ticket_id="request-c:attempt:0",
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            baseline=self.baseline,
            candidates=(self.small, self.large),
            output_tokens=30,
            context_length=4096,
            active_batch=1,
            deadline_us=1_000_000,
            slot_id=7,
            first_token_index=1,
            first_token_at_us=1_000,
            config=self.config,
            helper_evidence_state="LEARNING",
        )

        reused = controller.snapshot("request-c")[
            "probe_fractions_ppm"
        ]
        self.assertEqual(len(reused), 1)
        self.assertIn(
            reused[0],
            {self.small.split_fraction_ppm, self.large.split_fraction_ppm},
        )
        reused_winner = next(
            policy for policy in (self.small, self.large)
            if policy.split_fraction_ppm == reused[0]
        )

        self.assertEqual(directive.state, "EXPLOITING")
        self.assertIsNotNone(directive.control)
        self.assertEqual(
            directive.control.policy.policy_hash,
            reused_winner.policy_hash,
        )
        directive = controller.acknowledge(
            "request-c",
            AdaptiveDecodePolicyAck(
                request_id="request-c",
                slot_id=7,
                plan_generation=directive.control.plan_generation,
                applied_token_index=1,
                applied_at_us=1_001,
                policy_hash=directive.control.policy.policy_hash,
            ),
        )

        at_us = 1_001
        for _ in range(20):
            if controller.snapshot("request-c")["state"] == "EXPLOITING":
                break
            target = controller.snapshot("request-c")["target_token"]
            self.assertIsNotNone(target)
            at_us += 2_000
            directive = controller.boundary(
                "request-c",
                slot_id=7,
                token_index=target,
                at_us=at_us,
            )
            self.assertIsNotNone(directive.boundary)
            boundary = directive.boundary
            directive = controller.record_window(
                "request-c",
                boundary,
                AdaptiveDecodeRawWindowObservation(
                    fleet_energy_uj_by_domain={
                        "fleet": 1_000_000 * boundary.token_count
                    },
                    phone_compute_us=0,
                    usb_transfer_us=0,
                    rpc_us=0,
                    exposed_tail_us=0,
                    output_valid=True,
                    evidence_ids=("synthetic:diagnostic-window",),
                    energy_boundary_id="synthetic-fleet",
                    energy_attribution_kind="diagnostic",
                ),
            )
            if directive.control is not None:
                at_us += 1
                directive = controller.acknowledge(
                    "request-c",
                    AdaptiveDecodePolicyAck(
                        request_id="request-c",
                        slot_id=7,
                        plan_generation=directive.control.plan_generation,
                        applied_token_index=target,
                        applied_at_us=at_us,
                        policy_hash=directive.control.policy.policy_hash,
                    ),
                )
        else:
            self.fail("diagnostic verification did not converge")

        self.assertEqual(
            controller.snapshot("request-c")["current_policy_hash"],
            reused_winner.policy_hash,
        )

    def test_persisted_winner_is_scoped_to_helper_layout_geometry(self):
        _, observed = self._run(40)
        geometry_a = "sha256:" + "a" * 64
        geometry_b = "sha256:" + "b" * 64
        observed = replace(
            observed,
            helper_layout_geometry_sha256=geometry_a,
        )
        body = {
            "groups": [observed.to_json()],
            "schema": ADAPTIVE_OBSERVATION_STORE_SCHEMA,
        }

        def probe_fractions(geometry: str) -> tuple[int, ...]:
            controller = AdaptiveDecodeController()
            controller.load_observations({
                **body,
                "store_sha256": canonical_sha256(body),
            })
            controller.start(
                request_id="request-layout-" + geometry[-1],
                ticket_id=(
                    "request-layout-" + geometry[-1] + ":attempt:0"
                ),
                model_artifact_sha256=ARTIFACT,
                planning_profile_sha256=PLAN,
                baseline=self.baseline,
                candidates=(self.small, self.large),
                output_tokens=30,
                context_length=64,
                active_batch=1,
                deadline_us=1_000_000,
                slot_id=7,
                first_token_index=1,
                first_token_at_us=1_000,
                config=self.config,
                helper_layout_generation=1,
                helper_layout_geometry_sha256=geometry,
            )
            return controller.snapshot(
                "request-layout-" + geometry[-1]
            )["probe_fractions_ppm"]

        self.assertEqual(len(probe_fractions(geometry_a)), 1)
        self.assertEqual(len(probe_fractions(geometry_b)), 2)
        restored = type(observed).from_json(observed.to_json())
        self.assertEqual(
            restored.helper_layout_geometry_sha256, geometry_a
        )

    def test_grouped_fraction_evidence_produces_route_estimate(self):
        _, first = self._run(40)
        previous = "0" * 64
        windows = []
        for row in first.windows:
            acknowledgement = (
                None
                if row.applied_ack is None
                else replace(row.applied_ack, request_id="request-b")
            )
            cloned = replace(
                row,
                request_id="request-b",
                applied_ack=acknowledgement,
                previous_record_sha256=previous,
            )
            windows.append(cloned)
            previous = cloned.record_sha256.removeprefix("sha256:")
        second = replace(
            first,
            request_id="request-b",
            ticket_id="request-b:attempt:0",
            windows=tuple(windows),
        )
        body = {
            "groups": [first.to_json(), second.to_json()],
            "schema": ADAPTIVE_OBSERVATION_STORE_SCHEMA,
        }
        controller = AdaptiveDecodeController()
        controller.load_observations({
            **body,
            "store_sha256": canonical_sha256(body),
        })

        estimate = controller.historical_route_estimate(
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            baseline=self.baseline,
            candidates=(self.small, self.large),
            context_length=64,
            active_batch=1,
            config=self.config,
        )

        self.assertIsNotNone(estimate)
        self.assertFalse(estimate.selected_policy.baseline)
        self.assertEqual(estimate.baseline_group_count, 2)
        self.assertEqual(estimate.selected_group_count, 2)
        self.assertEqual(estimate.energy_boundary_id, "synthetic-fleet")
        self.assertLess(
            estimate.selected_energy_upper_per_token_uj,
            estimate.baseline_energy_lower_per_token_uj,
        )

        compatible_shape = controller.historical_route_estimate(
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            baseline=self.baseline,
            candidates=(self.small, self.large),
            context_length=4096,
            active_batch=1,
            config=self.config,
        )
        self.assertIsNotNone(compatible_shape)
        self.assertEqual(
            compatible_shape.requested_context_bucket,
            (4096).bit_length(),
        )
        self.assertEqual(
            compatible_shape.evidence_context_bucket,
            (64).bit_length(),
        )

        hot_baseline = replace(
            self.baseline,
            route_id="desktop-parent:hot",
            desktop_parent_route_id="desktop-parent:hot",
        )
        hot_candidates = tuple(
            replace(
                policy,
                route_id=policy.route_id + ":hot",
                desktop_parent_route_id=hot_baseline.route_id,
            )
            for policy in (self.small, self.large)
        )
        hot_estimate = controller.historical_route_estimate(
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            baseline=hot_baseline,
            candidates=hot_candidates,
            context_length=64,
            active_batch=1,
            config=self.config,
        )

        self.assertIsNotNone(hot_estimate)
        self.assertEqual(
            hot_estimate.selected_policy.split_fraction_ppm,
            estimate.selected_policy.split_fraction_ppm,
        )

    def test_assumed_power_can_reuse_matching_route_geometry(self):
        _, first = self._run(40)

        def clone_group(group, request_id):
            previous = "0" * 64
            windows = []
            for row in group.windows:
                acknowledgement = (
                    None
                    if row.applied_ack is None
                    else replace(
                        row.applied_ack, request_id=request_id
                    )
                )
                cloned = replace(
                    row,
                    request_id=request_id,
                    applied_ack=acknowledgement,
                    fleet_energy_uj_by_domain={
                        **dict(row.fleet_energy_uj_by_domain),
                        "phone-system": 1,
                    },
                    previous_record_sha256=previous,
                )
                windows.append(cloned)
                previous = cloned.record_sha256.removeprefix("sha256:")
            return replace(
                group,
                request_id=request_id,
                ticket_id=request_id + ":attempt:0",
                planning_profile_sha256="sha256:" + "4" * 64,
                windows=tuple(windows),
            )

        groups = (
            clone_group(first, "legacy-a"),
            clone_group(first, "legacy-b"),
        )
        body = {
            "groups": [group.to_json() for group in groups],
            "schema": ADAPTIVE_OBSERVATION_STORE_SCHEMA,
        }
        controller = AdaptiveDecodeController()
        controller.load_observations({
            **body,
            "store_sha256": canonical_sha256(body),
        })
        current_baseline = replace(
            self.baseline,
            route_id="current-desktop",
            executor_id="current-desktop-executor",
            operator_plan_sha256="sha256:" + "5" * 64,
            desktop_parent_route_id="current-desktop",
            resource_ids=("current-cpu", "current-gpu"),
        )
        current_candidates = tuple(
            replace(
                row,
                route_id="current-" + row.route_id,
                executor_id="current-phone-executor",
                operator_plan_sha256="sha256:" + "6" * 64,
                desktop_parent_route_id=current_baseline.route_id,
                resource_ids=(
                    "current-cpu", "current-gpu", "current-phone",
                ),
            )
            for row in (self.small, self.large)
        )
        arguments = {
            "model_artifact_sha256": ARTIFACT,
            "planning_profile_sha256": "sha256:" + "7" * 64,
            "component_capability_sha256": "sha256:" + "8" * 64,
            "baseline": current_baseline,
            "candidates": current_candidates,
            "context_length": 64,
            "active_batch": 1,
            "config": self.config,
            "phone_power_by_domain": {"phone-system": (1, 1)},
        }

        self.assertIsNone(
            controller.historical_route_estimate_with_assumed_phone_power(
                **arguments
            )
        )
        estimate = (
            controller.historical_route_estimate_with_assumed_phone_power(
                **arguments,
                route_geometry_prior=True,
            )
        )

        self.assertIsNotNone(estimate)
        self.assertEqual(estimate.evidence_match, "route_geometry_prior")
        self.assertFalse(estimate.selected_policy.baseline)
        self.assertLess(
            estimate.selected_energy_upper_per_token_uj,
            estimate.baseline_energy_lower_per_token_uj,
        )

    def test_assumed_power_reuses_a_validated_operator_superset(self):
        _, first = self._run(40)
        target_component = "sha256:" + "8" * 64

        def clone_group(group, request_id):
            previous = "0" * 64
            windows = []
            for row in group.windows:
                acknowledgement = (
                    None
                    if row.applied_ack is None
                    else replace(
                        row.applied_ack, request_id=request_id
                    )
                )
                cloned = replace(
                    row,
                    request_id=request_id,
                    applied_ack=acknowledgement,
                    fleet_energy_uj_by_domain={
                        **dict(row.fleet_energy_uj_by_domain),
                        "phone-system": 1,
                    },
                    previous_record_sha256=previous,
                )
                windows.append(cloned)
                previous = cloned.record_sha256.removeprefix("sha256:")
            return replace(
                group,
                request_id=request_id,
                ticket_id=request_id + ":attempt:0",
                windows=tuple(windows),
            )

        groups = (
            clone_group(first, "subset-a"),
            clone_group(first, "subset-b"),
        )
        body = {
            "groups": [group.to_json() for group in groups],
            "schema": ADAPTIVE_OBSERVATION_STORE_SCHEMA,
        }
        controller = AdaptiveDecodeController()
        controller.load_observations({
            **body,
            "store_sha256": canonical_sha256(body),
        })
        source = self.large
        subset = replace(
            source,
            route_id="phone-large-subset",
            layer_indices=(2,),
            layer_mask=1 << 2,
        )
        arguments = {
            "model_artifact_sha256": ARTIFACT,
            "planning_profile_sha256": PLAN,
            "component_capability_sha256": target_component,
            "baseline": self.baseline,
            "candidates": (subset,),
            "context_length": 64,
            "active_batch": 1,
            "config": self.config,
            "phone_power_by_domain": {"phone-system": (1, 1)},
        }

        self.assertIsNone(
            controller.historical_route_estimate_with_assumed_phone_power(
                **arguments
            )
        )
        with self.assertRaises(AdaptiveDecodeError):
            controller.historical_route_estimate_with_assumed_phone_power(
                **arguments,
                operator_subset_prior=True,
            )
        estimate = (
            controller.historical_route_estimate_with_assumed_phone_power(
                **arguments,
                operator_subset_prior=True,
                operator_subset_source_layer_mask=source.layer_mask,
            )
        )

        self.assertIsNotNone(estimate)
        self.assertEqual(
            estimate.evidence_match,
            "route_operator_subset_prior",
        )
        self.assertEqual(estimate.source_layer_mask, source.layer_mask)
        self.assertEqual(estimate.target_layer_mask, subset.layer_mask)
        self.assertEqual(estimate.evidence_scale_ppm, 500_000)
        self.assertLess(
            estimate.selected_energy_upper_per_token_uj,
            estimate.baseline_energy_lower_per_token_uj,
        )
        self.assertIsNone(
            controller.historical_route_estimate_with_assumed_phone_power(
                **arguments,
                operator_subset_prior=True,
                operator_subset_source_layer_mask=1 << 7,
            )
        )

    def test_one_compatible_request_limits_the_next_probe_to_its_winner(self):
        _, first = self._run(40)
        body = {
            "groups": [first.to_json()],
            "schema": ADAPTIVE_OBSERVATION_STORE_SCHEMA,
        }
        controller = AdaptiveDecodeController()
        controller.load_observations({
            **body,
            "store_sha256": canonical_sha256(body),
        })

        controller.start(
            request_id="request-b",
            ticket_id="request-b:attempt:0",
            model_artifact_sha256=ARTIFACT,
            planning_profile_sha256=PLAN,
            baseline=self.baseline,
            candidates=(self.small, self.large),
            output_tokens=18,
            context_length=64,
            active_batch=1,
            deadline_us=1_000_000,
            slot_id=7,
            first_token_index=1,
            first_token_at_us=1_000,
            config=replace(
                self.config,
                maximum_latency_ppm=1_250_000,
            ),
            ticket_policy=first.final_policy,
        )

        self.assertEqual(
            controller.snapshot("request-b")["probe_fractions_ppm"],
            (first.final_policy.split_fraction_ppm,),
        )

    def test_window_decision_time_is_below_five_milliseconds(self):
        controller = AdaptiveDecodeController()
        directive = self.start(controller)
        directive = self.record(
            controller, directive.target_token_index, 3_000, 100
        )
        self.acknowledge(controller, directive.control, 3, 3_001)
        timing = controller.timing("request-a")
        self.assertLess(timing["maximum_us"], 5_000)


if __name__ == "__main__":
    unittest.main()
