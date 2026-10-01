"""Adaptive decode control: start, boundaries, windows, acknowledgement, observations.

Mixin of ``UnifiedScheduler``; methods were moved here verbatim and rely on the
state initialised in ``UnifiedScheduler.__init__``.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Mapping, NamedTuple
from .._internal.lifecycle import UnifiedScheduleError
from .._internal.runtime_capabilities import RuntimeCapabilityCatalog
from .._internal.model_placement_controller import ModelPlacementControllerError
from .._internal.phone_shards import PhoneShardPlacementError, artifact_layout_identity_sha256
from .._internal.runtime_controller import RuntimeControllerError, RuntimeRequestTicket
from .._internal.plan_contracts.co_helpers import co_helper_declaration
from .._internal.adaptive_decode_ops.coherence import INHERITED_REASON
from .._internal.types import canonical_sha256
from .._internal.adaptive_decode_contracts import (
    AdaptiveDecodeConfig,
    AdaptiveDecodeControl,
    AdaptiveDecodeDirective,
    AdaptiveDecodeError,
    AdaptiveDecodeGroupedObservation,
    AdaptiveDecodePolicy,
    AdaptiveDecodePolicyAck,
    AdaptiveDecodeRawWindowObservation,
    AdaptiveDecodeWindowBoundary,
    policy_device_set,
)
from .automated_requests_ops import failure as _failure
from .automated_requests_ops.event_replanning import (
    event_replanning_enabled,
    note_device_admissible,
    note_resource_release,
)
from .common import (
    _runtime_serialized,
    _text,
)


DEVICE_ABSENT_AT_START = "DEVICE_ABSENT_AT_START"
DEVICE_MEMBERSHIP_EVENT_KINDS = (
    DEVICE_ABSENT_AT_START, "DEVICE_QUARANTINED", "DEVICE_READMITTED",
)


def device_quarantine_resource_ids(catalog: RuntimeCapabilityCatalog, device_id: str) -> tuple[str, ...]:
    """Resources only this phone owns. A static co-helper quarantines none: its resources also bind
    the composite that serves the primary-only device sets, which the adaptive controller keeps
    off the lost phone instead."""
    executor = catalog.executor_by_device.get(device_id)
    if executor is None or any(
        device_id in declaration.device_ids
        for declaration in (
            co_helper_declaration(row.adapter_parameters) for row in catalog.composite_executors
        )
        if declaration is not None
    ):
        return ()
    shared = {
        resource_id
        for row in catalog.executors if row.device_id != device_id
        for resource_id in row.execution_resource_ids
    }
    return tuple(sorted(set(executor.execution_resource_ids) - shared))


class _AdaptiveStartHelperState(NamedTuple):
    ready: bool
    layout_generation: int | None
    layout_geometry_sha256: str | None
    layout_identity_sha256: str | None


def _phone_device(endpoint: str) -> str:
    """session://<phone>/<session> -> session://<phone>; other forms name only themselves."""
    return endpoint.rpartition("/")[0] or endpoint


class AdaptiveDecodeControlMixin:
    """Adaptive decode control: start, boundaries, windows, acknowledgement, observations."""

    def _membership_events(self) -> list[dict[str, object]]:
        events = getattr(self, "_device_membership_event_rows", None)
        if events is None:
            events = self._device_membership_event_rows = []
        return events

    def _membership_device(self, device_id: object) -> str:
        device_id = _text("membership device id", device_id)
        catalog = self._runtime_capabilities
        if catalog is None:
            raise UnifiedScheduleError("device membership requires registered runtime capabilities")
        device = catalog.placement_profile.devices.get(device_id)
        if device is None or device.kind != "phone" or device_id not in catalog.executor_by_device:
            raise UnifiedScheduleError("membership device is not a phone of the runtime catalog")
        return device_id

    @staticmethod
    def _membership_time(at_us: object) -> int:
        if type(at_us) is not int or at_us < 0:
            raise UnifiedScheduleError("membership time is invalid")
        return at_us

    @_runtime_serialized
    def quarantine_device(self, device_id: str, *, reason: str, at_us: int) -> None:
        """Take one phone out of placement (lost at runtime or absent at start); idempotent.

        Every server-policy group drops its device sets, live sessions eliminate its policies and
        the runtime controller withholds the resources only it owns until ``readmit_device``."""
        device_id = self._membership_device(device_id)
        reason = _text("membership quarantine reason", reason)
        at_us = self._membership_time(at_us)
        resources = device_quarantine_resource_ids(self._runtime_capabilities, device_id)
        try:
            adaptive = self._adaptive_decode.quarantine_device(device_id, reason=reason, at_us=at_us)
            runtime = self._runtime_controller.quarantine_device(
                device_id, resource_ids=resources, reason=reason, at_us=at_us)
        except (AdaptiveDecodeError, RuntimeControllerError) as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        if adaptive or runtime:
            self._membership_events().append({
                "at_us": at_us,
                "device_id": device_id,
                "kind": DEVICE_ABSENT_AT_START if reason == DEVICE_ABSENT_AT_START else "DEVICE_QUARANTINED",
                "reason": reason,
            })

    @_runtime_serialized
    def readmit_device(self, device_id: str, *, at_us: int, identity_sha256: str) -> None:
        """Return a rejoined phone whose pinned identity the rig verified; idempotent. Its device
        sets become eligible again and the normal probe decides on new evidence."""
        device_id = self._membership_device(device_id)
        at_us = self._membership_time(at_us)
        identity_sha256 = _text("membership identity", identity_sha256)
        if (not identity_sha256.startswith("sha256:") or len(identity_sha256) != 71
                or set(identity_sha256[7:]) - set("0123456789abcdef")):
            raise UnifiedScheduleError("membership identity must be sha256:<64 hex>")
        quarantined_at_us = next((
            row.get("at_us") for row in reversed(self._membership_events())
            if row.get("device_id") == device_id and row.get("kind") != "DEVICE_READMITTED"
        ), None) if event_replanning_enabled(self) else None
        try:
            adaptive = self._adaptive_decode.readmit_device(device_id)
            runtime = self._runtime_controller.readmit_device(device_id)
        except (AdaptiveDecodeError, RuntimeControllerError) as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        if adaptive or runtime:
            self._membership_events().append({
                "at_us": at_us,
                "device_id": device_id,
                "identity_sha256": identity_sha256,
                "kind": "DEVICE_READMITTED",
                "reason": "IDENTITY_VERIFIED_JOIN",
            })
            note_device_admissible(
                self, device_id, at_us, "DEVICE_READMITTED", quarantined_at_us
            )

    def quarantined_devices(self) -> Mapping[str, str]:
        """Quarantined phone -> reason (the controller's own lock: rig probes poll it)."""
        return dict(self._adaptive_decode.quarantined_devices)

    def device_membership_events(self) -> tuple[Mapping[str, object], ...]:
        """RESULT ``device_membership_events`` rows, in order."""
        with self._runtime_lock:
            return tuple(dict(row) for row in self._membership_events())

    def register_joint_join_prefill_yield(
        self, ticket: RuntimeRequestTicket, *, at_us: int, expires_at_us: int,
    ) -> Mapping[str, object]:
        """``dispatch_policy.joint_planner`` (active): the planner joins an acquired request into its
        running server, so the co-tenants of its model and desktop parent run the host policy until
        its decode starts (``_internal.adaptive_decode_ops.prefill_yield``). Called from the journal
        hook under the runtime lock; touches only the adaptive controller (a leaf lock). Registers
        nothing, and says why, when the join cannot be expressed through server policy coherence."""
        config = getattr(self, "_adaptive_decode_config", None)
        plan = ticket.execution_plan
        if config is None or not config.server_policy_coherence:
            return {"outcome": "SERVER_POLICY_COHERENCE_DISABLED"}
        if ticket.dispatch_state != "ACQUIRED":
            return {"outcome": "JOINER_NOT_ACQUIRED"}
        if plan is None or plan.desktop_placement_sha256 is None:
            return {"outcome": "JOINER_WITHOUT_DESKTOP_PARENT"}
        if (plan.execution_contract.execution_mode != "adaptive-split"
                and plan.helper_envelope is None and not self._has_dormant_phone_ffn_runtime(plan)):
            return {"outcome": "JOINER_NOT_ADAPTIVE"}
        try:
            outcome, co_tenants = self._adaptive_decode.register_prefill_yield(
                ticket.request.request_id, model_artifact_sha256=ticket.model.artifact_sha256,
                desktop_placement_sha256=plan.desktop_placement_sha256, at_us=at_us,
                expires_at_us=expires_at_us)
        except AdaptiveDecodeError as exc:
            return {"outcome": "REGISTRATION_REJECTED", "detail": str(exc)}
        return {"outcome": outcome, "co_tenant_request_ids": list(co_tenants)}

    def clear_joint_join_prefill_yield(self, request_id: str, *, at_us: int, reason: str) -> bool:
        """End a joiner's prefill yield (it finished, failed or was cancelled before decoding)."""
        return self._adaptive_decode.clear_prefill_yield(request_id, at_us=at_us, reason=reason)

    def joint_join_prefill_yield_events(self) -> tuple[Mapping[str, object], ...]:
        return self._adaptive_decode.prefill_yield_events()

    def _adaptive_policies_from_ticket(
        self,
        ticket: RuntimeRequestTicket,
    ) -> tuple[
        AdaptiveDecodePolicy,
        tuple[AdaptiveDecodePolicy, ...],
        AdaptiveDecodePolicy | None,
    ]:
        late = self._materialize_late_request_helper(ticket)
        if late is not None:
            return late.baseline, late.candidates, late.ticket_policy
        plan = ticket.execution_plan
        if plan is None:
            raise UnifiedScheduleError(
                "adaptive ticket lacks its execution plan"
            )
        if (
            plan.helper_envelope is None
            and self._has_dormant_phone_ffn_runtime(plan)
        ):
            baseline, _ticket_candidates = (
                self._ticket_adaptive_policy_records(ticket)
            )
            opportunity = self._request_helper_opportunity_for_ticket(
                ticket
            )
            if opportunity is None:
                return baseline, (), None
            policy_rows = self._helper_opportunity_policies(
                ticket, opportunity
            )
            if policy_rows is None:
                return baseline, (), None
            baseline, candidates, _ticket_policy = policy_rows
            return baseline, candidates, None
        baseline, candidates = self._ticket_adaptive_policy_records(ticket)
        ticket_policy = None
        helper = plan.helper_envelope
        selected_route_id = (
            helper.route_id
            if helper is not None else ticket.decision.route_id
        )
        if selected_route_id != baseline.route_id:
            selected = tuple(
                row for row in candidates
                if row.route_id == selected_route_id
            )
            if len(selected) != 1:
                raise UnifiedScheduleError(
                    "adaptive ticket lacks its selected envelope"
                )
            envelope = selected[0]
            ticket_policy = envelope
            helper_plan = (
                plan if helper is None else helper.helper_plan
            )
            parameters = helper_plan.adapter_parameters
            resident_layer_mask = parameters.get(
                "ffn_resident_layer_mask"
            )
            resident_columns = parameters.get("ffn_resident_columns")
            if (
                type(resident_layer_mask) is not int
                or resident_layer_mask <= 0
                or type(resident_columns) is not int
                or resident_columns <= 0
            ):
                raise UnifiedScheduleError(
                    "adaptive ticket lacks resident FFN geometry"
                )
            candidates = tuple(
                row for row in candidates
                if row.executor_id == envelope.executor_id
                and row.operator_plan_sha256
                    == helper_plan.plan_sha256
                and row.desktop_parent_route_id
                    == envelope.desktop_parent_route_id
                and row.desktop_placement_sha256
                    == envelope.desktop_placement_sha256
                and row.layer_mask & ~resident_layer_mask == 0
                and row.columns <= resident_columns
            )
        else:
            candidates = ()
        leased = {lease.resource_id for lease in ticket.decision.leases}
        if not set(baseline.resource_ids).issubset(leased):
            raise UnifiedScheduleError(
                "adaptive desktop resources are not covered by the ticket"
            )
        if helper is None and any(
            not set(row.resource_ids).issubset(leased)
            for row in candidates
        ):
            raise UnifiedScheduleError(
                "adaptive policy resources are not covered by the ticket"
            )
        contract = (
            plan.execution_contract
            if helper is None else helper.helper_plan.execution_contract
        )
        allowed = set(contract.allowed_adaptive_fractions_ppm)
        if (
            contract.execution_mode != "adaptive-split"
            or contract.initial_split_fraction_ppm
                != baseline.split_fraction_ppm
            or any(
                row.split_fraction_ppm not in allowed
                for row in candidates
            )
        ):
            raise UnifiedScheduleError(
                "adaptive policies differ from the execution contract"
            )
        return baseline, candidates, ticket_policy

    @_runtime_serialized
    def start_adaptive_decode(
        self,
        request_id: str,
        *,
        slot_id: int,
        first_token_index: int,
        at_us: int,
        context_length: int | None = None,
        active_batch: int = 1,
        config: AdaptiveDecodeConfig | None = None,
        execution_context_available: bool = True,
    ) -> AdaptiveDecodeDirective:
        """Open scheduler-owned token windows for an acquired adaptive ticket."""
        ticket = self.runtime_execution_ticket(request_id)
        self._release_restarted_adaptive_attempt(request_id, ticket.ticket_id)
        plan = ticket.execution_plan
        dormant = bool(
            plan is not None
            and self._has_dormant_phone_ffn_runtime(plan)
        )
        ticket_opportunity = (
            self._request_helper_opportunity_for_ticket(ticket)
        )
        dormant_opportunity = (
            None if not dormant else ticket_opportunity
        )
        if plan is None or (
            plan.execution_contract.execution_mode != "adaptive-split"
            and plan.helper_envelope is None
            and not dormant
        ):
            raise UnifiedScheduleError(
                "runtime ticket is not an adaptive decode request"
            )
        baseline, candidates, ticket_policy = (
            self._adaptive_policies_from_ticket(ticket)
        )
        if dormant and not candidates:
            dormant_opportunity = None
        try:
            helper = self._request_helper_envelope(ticket)
            late = self._late_request_helper_contexts.get(request_id)
            helper_state = self._adaptive_start_helper_state(
                ticket,
                plan,
                helper,
                slot_id=slot_id,
                first_token_index=first_token_index,
                at_us=at_us,
            )
            selected_route_id = (
                helper.route_id
                if helper is not None else
                dormant_opportunity.route_id
                if dormant_opportunity is not None else
                ticket.decision.route_id
            )
            selected_estimate, dormant_opportunity = (
                self._adaptive_start_selected_estimate(
                    ticket,
                    selected_route_id,
                    helper,
                    late,
                    dormant_opportunity,
                )
            )
            transition_cost_us, transition_energy_uj = (
                self._adaptive_start_transition_costs(selected_estimate)
            )
            allow_assumed_phone_power = (
                self._adaptive_start_allow_assumed_phone_power(
                    plan, helper, dormant_opportunity
                )
            )
            directive = self._adaptive_decode.start(
                request_id=request_id,
                ticket_id=ticket.ticket_id,
                model_artifact_sha256=ticket.model.artifact_sha256,
                planning_profile_sha256=(
                    ticket.planning_profile_sha256
                ),
                component_capability_sha256=(
                    self._adaptive_start_component_capability_sha256(
                        ticket,
                        plan,
                        helper,
                        late,
                        dormant_opportunity,
                    )
                ),
                baseline=baseline,
                candidates=candidates,
                output_tokens=ticket.request.output_tokens,
                context_length=(
                    ticket.request.input_tokens
                    if context_length is None else context_length
                ),
                active_batch=active_batch,
                deadline_us=ticket.request.deadline_us,
                slot_id=slot_id,
                first_token_index=first_token_index,
                first_token_at_us=at_us,
                config=(
                    self._adaptive_start_config(
                        transition_cost_us=transition_cost_us,
                        transition_energy_uj=transition_energy_uj,
                        allow_assumed_phone_power=allow_assumed_phone_power,
                    )
                    if config is None else config
                ),
                ticket_policy=ticket_policy,
                helper_available=helper_state.ready,
                helper_layout_generation=helper_state.layout_generation,
                helper_layout_geometry_sha256=(
                    helper_state.layout_geometry_sha256
                ),
                helper_layout_identity_sha256=helper_state.layout_identity_sha256,
                helper_evidence_state=(
                    late.evidence_state
                    if late is not None else
                    dormant_opportunity.evidence_state
                    if dormant_opportunity is not None else
                    ticket_opportunity.evidence_state
                    if ticket_opportunity is not None else
                    "TRUSTED"
                ),
                phone_power_policy_from_capability=config is None,
                execution_context_available=execution_context_available,
            )
            self._report_helper_phone_disturbance(request_id)
            return self._track_adaptive_directive(
                request_id,
                directive,
                at_us=at_us,
                token_index=first_token_index,
            )
        except AdaptiveDecodeError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    def _release_restarted_adaptive_attempt(
        self, request_id: str, current_ticket_id: str
    ) -> None:
        """Elastic phones: a recovered request's new attempt replaces the adaptive registration an
        earlier attempt of the SAME request left when it failed with helper_lost / server_exited
        (a FALLBACK and replans later). Only tickets recorded as such failures qualify, never the
        current ticket or another request; each release is kept in
        ``adaptive_decode_restarted_attempts``. Without such a failure nothing changes."""
        stale = _failure.elastic_failed_attempts(self, request_id) - {current_ticket_id}
        if not stale:
            return
        try:
            row = self._adaptive_decode.release_restarted_attempt(
                request_id, tuple(sorted(stale)), "ELASTIC_ATTEMPT_RESTARTED"
            )
        except AdaptiveDecodeError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        if row is not None:
            if not hasattr(self, "_adaptive_restarted_attempts"):
                self._adaptive_restarted_attempts = []
            self._adaptive_restarted_attempts.append(
                {**row, "current_ticket_id": current_ticket_id}
            )

    @_runtime_serialized
    def adaptive_decode_restarted_attempts(self) -> tuple[Mapping[str, object], ...]:
        """Adaptive registrations of failed earlier attempts that a recovered attempt replaced."""
        return tuple(
            MappingProxyType(dict(row))
            for row in getattr(self, "_adaptive_restarted_attempts", ())
        )

    def _adaptive_start_helper_state(
        self,
        ticket: RuntimeRequestTicket,
        plan,
        helper,
        *,
        slot_id: int,
        first_token_index: int,
        at_us: int,
    ) -> _AdaptiveStartHelperState:
        helper_ready = False
        helper_generation = None
        helper_geometry = None
        helper_identity = None
        if helper is not None:
            helper_ready = self._ready_request_helper(ticket, helper) is not None
            if helper_ready:
                helper_generation = helper.phone_layout_generation
                helper_geometry = (
                    helper.phone_layout_geometry_sha256
                )
                helper_identity = self._adaptive_layout_identity_sha256(
                    ticket, helper_generation
                )
        elif plan.execution_contract.execution_mode == "adaptive-split":
            helper_ready = True
        return _AdaptiveStartHelperState(
            helper_ready, helper_generation, helper_geometry, helper_identity
        )

    def _adaptive_layout_identity_sha256(
        self, ticket: RuntimeRequestTicket, generation: int,
    ) -> str | None:
        """The request model's own phone shards in one layout generation, which key the
        coherent server policy; None (a group per generation) without coherence or layout."""
        config = getattr(self, "_adaptive_decode_config", None)
        if config is None or not config.server_policy_coherence:
            return None
        try:
            layout = self._model_placement_controller.phone_layout(generation)
            return artifact_layout_identity_sha256(layout.layout, ticket.model.artifact_sha256)
        except (ModelPlacementControllerError, PhoneShardPlacementError):
            return None

    @staticmethod
    def _adaptive_start_selected_estimate(
        ticket: RuntimeRequestTicket,
        selected_route_id: str,
        helper,
        late,
        dormant_opportunity,
    ):
        selected_estimate = next((
            row for row in ticket.cost_estimates.estimates
            if row.route_id == selected_route_id
        ), None)
        if late is not None or (
            selected_estimate is None and helper is None
        ):
            # READY helpers carry current marginal costs. A matching cold
            # ticket estimate still includes the already-paid shard load.
            dormant_opportunity = None
            selected_estimate = next((
                row for row in ticket.cost_estimates.estimates
                if row.route_id == ticket.decision.route_id
            ), None)
        if selected_estimate is None:
            raise UnifiedScheduleError(
                "adaptive ticket lacks its selected cost estimate"
            )
        return selected_estimate, dormant_opportunity

    @staticmethod
    def _adaptive_start_transition_costs(selected_estimate) -> tuple[int, int]:
        break_even = selected_estimate.details.get(
            "residency_break_even"
        )
        transition_cost_us = 2_000
        transition_energy_uj = 20_000
        if isinstance(break_even, Mapping):
            transition_cost_us = max(
                1,
                int(break_even.get(
                    "incremental_transition_latency_us",
                    transition_cost_us,
                )),
            )
            transition_energy_uj = max(
                1,
                int(break_even.get(
                    "incremental_transition_energy_uj",
                    transition_energy_uj,
                )),
            )
        return transition_cost_us, transition_energy_uj

    def _adaptive_start_allow_assumed_phone_power(
        self,
        plan,
        helper,
        dormant_opportunity,
    ) -> bool:
        execution_contract = (
            plan.execution_contract
            if helper is None
            else helper.helper_plan.execution_contract
        )
        phone_device_id = (
            execution_contract.phone_device_id
            if execution_contract.phone_device_id is not None
            else None
            if dormant_opportunity is None else
            dormant_opportunity.helper_operator_plan
                .execution_contract.phone_device_id
        )
        phone_power_profile = (
            None
            if phone_device_id is None
            else self._runtime_capabilities
                .phone_power_profile_by_device.get(phone_device_id)
        )
        return (
            phone_power_profile is not None
            and phone_power_profile.allow_assumed_for_scheduling
        )

    def _adaptive_start_component_capability_sha256(
        self,
        ticket: RuntimeRequestTicket,
        plan,
        helper,
        late,
        dormant_opportunity,
    ) -> str:
        if late is not None:
            return late.component.identity_sha256
        return self._automated_compiler().component_capability_identity(
            (
                dormant_opportunity.helper_operator_plan
                if dormant_opportunity is not None else
                plan if helper is None else
                helper.helper_plan
            ),
            (
                next(
                    row.executor_id
                    for row in ticket.executor_bindings
                    if row.route_id
                        == dormant_opportunity.route_id
                )
                if dormant_opportunity is not None else
                ticket.binding.executor_id
                if helper is None else
                helper.helper_binding.executor_id
            ),
        )

    def _record_adaptive_helper_energy_policy_binding(
        self, ticket, helper, previous_state, at_us: int,
    ) -> None:
        key = "allow_assumed_phone_power_for_operational_selection"
        allowed = self._adaptive_decode.snapshot(ticket.request.request_id)[key]
        if allowed == previous_state[key]:
            return
        self._model_placement_controller.record_request_helper_event(
            ticket.request.request_id, "HELPER_ENERGY_POLICY_BOUND", at_us,
            {
                "request_ticket_id": ticket.ticket_id,
                "phone_layout_generation": helper.phone_layout_generation,
                "phone_layout_geometry_sha256": helper.phone_layout_geometry_sha256,
                "operator_plan_sha256": helper.operator_plan_sha256,
                "allow_assumed_for_operational_selection": allowed,
                "qualification_state": "DIAGNOSTIC",
            },
        )

    def _adaptive_start_config(
        self,
        *,
        transition_cost_us: int,
        transition_energy_uj: int,
        allow_assumed_phone_power: bool,
    ) -> AdaptiveDecodeConfig:
        return replace(
            self._adaptive_decode_config,
            minimum_energy_saving_ppm=(
                self._runtime_capabilities
                .minimum_energy_saving_ppm
            ),
            maximum_latency_ppm=(
                self._runtime_capabilities.maximum_latency_ppm
            ),
            transition_cost_us=transition_cost_us,
            transition_energy_uj=transition_energy_uj,
            allow_assumed_phone_power_for_operational_selection=(
                allow_assumed_phone_power
            ),
        )

    def _track_adaptive_directive(
        self,
        request_id: str,
        directive: AdaptiveDecodeDirective | None,
        *,
        at_us: int,
        token_index: int,
    ) -> AdaptiveDecodeDirective | None:
        if directive is None:
            return directive
        self._record_assistance_decision(request_id, directive, token_index, at_us)
        if directive.reason.startswith("VERIFICATION_"):
            snapshot = self._adaptive_decode.snapshot(request_id)
            verification = snapshot["verification"]
            self._model_placement_controller.record_request_helper_event(
                request_id, directive.reason, at_us,
                {"token_index": token_index, "evidence_state": "DIAGNOSTIC",
                 "phone_layout_generation": snapshot["helper_layout_generation"],
                 "phone_layout_geometry_sha256": snapshot["helper_layout_geometry_sha256"],
                 **verification},
            )
        if directive.reason in {
            "INSUFFICIENT_OPPORTUNITY", "PROBE_INCOMPLETE", "PROBE_CANDIDATE_REJECTED",
        }:
            rejected = {}
            if directive.reason == "PROBE_CANDIDATE_REJECTED":
                rejected["eliminated_policy_reasons"] = self._adaptive_decode.snapshot(
                    request_id)["eliminated_policy_reasons"]
            self._model_placement_controller.record_request_helper_event(
                request_id, directive.reason, at_us,
                {"token_index": token_index, "evidence_state": "DIAGNOSTIC",
                 "accepted": False, "reason": directive.reason, **rejected},
            )
        ticket = self.runtime_execution_ticket(request_id)
        helper = (
            None
            if ticket.execution_plan is None
            else self._request_helper_envelope(ticket)
        )
        if directive.control is None:
            if helper is None or directive.target_token_index is None:
                return directive
            try:
                policy = self._adaptive_decode.active_policy(request_id)
            except AdaptiveDecodeError as exc:
                raise UnifiedScheduleError(str(exc)) from exc
            if policy is None or policy.baseline:
                return directive
            attempt = self._try_attach_ready_request_helper(
                ticket,
                slot_id=(
                    self._model_placement_controller
                    .request_binding(request_id)["base"]["server_slot_id"]
                ),
                token_index=token_index,
                at_us=at_us,
                fraction_ppm=policy.split_fraction_ppm,
            )
            if attempt.attached:
                return directive
            try:
                fallback = self._adaptive_decode.yield_helper_window(
                    request_id,
                    token_index=token_index,
                    at_us=at_us,
                )
            except AdaptiveDecodeError as exc:
                raise UnifiedScheduleError(str(exc)) from exc
            self._model_placement_controller.record_request_helper_event(
                request_id,
                "WINDOW_LEASE_YIELDED",
                at_us,
                {
                    "reason": attempt.reason,
                    "requested_fraction_ppm": (
                        policy.split_fraction_ppm
                    ),
                    "token_index": token_index,
                },
            )
            return self._track_adaptive_directive(
                request_id,
                fallback,
                at_us=at_us,
                token_index=token_index,
            )
        fraction_ppm = directive.control.policy.split_fraction_ppm
        if helper is not None:
            if fraction_ppm == 0:
                return directive
            attempt = self._try_attach_ready_request_helper(
                ticket,
                slot_id=directive.control.slot_id,
                token_index=token_index,
                at_us=at_us,
                fraction_ppm=fraction_ppm,
            )
            if not attempt.attached:
                self._model_placement_controller.record_request_helper_event(
                    request_id,
                    (
                        "EXECUTION_DEFERRED"
                        if attempt.retryable else "EXECUTION_REJECTED"
                    ),
                    at_us,
                    {
                        "phone_layout_generation": (
                            helper.phone_layout_generation
                        ),
                        "phone_layout_geometry_sha256": (
                            helper.phone_layout_geometry_sha256
                        ),
                        "reason": attempt.reason,
                        "requested_fraction_ppm": fraction_ppm,
                        "token_index": token_index,
                    },
                )
                try:
                    if attempt.retryable:
                        if attempt.reason == "PHONE_HELPER_NOT_READY":
                            self._adaptive_decode.helper_unavailable(
                                request_id
                            )
                        return self._adaptive_decode.defer_control(
                            request_id,
                            directive.control,
                            attempt.reason,
                            at_us=at_us,
                        )
                    fallback = self._adaptive_decode.control_failed(
                        request_id,
                        directive.control,
                        attempt.reason,
                        at_us=at_us,
                    )
                except AdaptiveDecodeError as exc:
                    raise UnifiedScheduleError(str(exc)) from exc
                return self._track_adaptive_directive(
                    request_id,
                    fallback,
                    at_us=at_us,
                    token_index=token_index,
                )
            return directive
        binding = self._model_placement_controller.request_binding(
            request_id
        )
        if binding is None:
            raise UnifiedScheduleError(
                "adaptive request lacks its placement binding"
            )
        bound_helper = binding.get("helper_envelope")
        if isinstance(bound_helper, Mapping):
            if fraction_ppm == 0:
                return directive
            self._model_placement_controller.record_request_helper_event(
                request_id,
                "EXECUTION_DEFERRED",
                at_us,
                {
                    "phone_layout_generation": (
                        bound_helper.get("phone_layout_generation")
                    ),
                    "phone_layout_geometry_sha256": (
                        bound_helper.get(
                            "phone_layout_geometry_sha256"
                        )
                    ),
                    "reason": "PHONE_HELPER_NOT_READY",
                    "requested_fraction_ppm": fraction_ppm,
                    "token_index": token_index,
                },
            )
            try:
                self._adaptive_decode.helper_unavailable(request_id)
                return self._adaptive_decode.defer_control(
                    request_id,
                    directive.control,
                    "PHONE_HELPER_NOT_READY",
                    at_us=at_us,
                )
            except AdaptiveDecodeError as exc:
                raise UnifiedScheduleError(str(exc)) from exc
        try:
            self._model_placement_controller.update_request_fraction(
                request_id,
                directive.control.policy.split_fraction_ppm,
                resident_component_identity_sha256=str(
                    binding["resident_component_identity_sha256"]
                ),
                observed_at_us=at_us,
            )
        except ModelPlacementControllerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        return directive

    def _record_late_helper_adoption(
        self, request_id: str, helper, *, token_index: int, at_us: int,
        source: str = "DECODE_BOUNDARY",
    ) -> None:
        """Record HELPER_ADOPTED_LATE when a helper-less session attaches a ready helper.

        Only under ``late_helper_adoption`` and only for a session that never had a
        helper (cj5: 003/005 decoded helper-less while their model's shards became
        READY 2-30 s after their start); sessions whose helper was ready from the
        start record nothing here. ``source`` names the path: the decode boundary,
        or the READY publication of the layout (s1a 001 adopted there).
        """
        if not self._adaptive_decode.adopts_late_helper(request_id, token_index=token_index):
            return
        self._model_placement_controller.record_request_helper_event(
            request_id,
            "HELPER_ADOPTED_LATE",
            at_us,
            {
                "phone_layout_generation": helper.phone_layout_generation,
                "phone_layout_geometry_sha256": helper.phone_layout_geometry_sha256,
                "ready_at_us": at_us,
                "request_id": request_id,
                "source": source,
                "token_index": token_index,
            },
        )

    @_runtime_serialized
    def adaptive_decode_boundary(
        self,
        request_id: str,
        *,
        slot_id: int,
        token_index: int,
        at_us: int,
        terminal: bool = False,
    ) -> AdaptiveDecodeDirective | None:
        try:
            ticket = self.runtime_execution_ticket(request_id)
            if not terminal:
                self._reevaluate_pending_phone_layout_at_boundary(
                    ticket, at_us
                )
            helper = (
                None
                if ticket.execution_plan is None
                else self._request_helper_envelope(ticket)
            )
            late = self._late_request_helper_contexts.get(request_id)
            placement_binding = (
                self._model_placement_controller.request_binding(
                    request_id
                )
            )
            attachment = (
                None if placement_binding is None else
                placement_binding.get("helper_attachment")
            )
            attached = bool(
                helper is not None
                and isinstance(attachment, Mapping)
                and attachment.get("phone_layout_generation")
                    == helper.phone_layout_generation
                and attachment.get("phone_layout_geometry_sha256")
                    == helper.phone_layout_geometry_sha256
                and attachment.get("operator_plan_sha256")
                    == helper.operator_plan_sha256
            )
            helper_rebind = (
                self._model_placement_controller
                .request_helper_rebind_state(request_id)
            )
            retains_helper_sessions = bool(
                helper_rebind is not None
                and tuple(helper_rebind.get(
                    "target_allowed_session_ids", ()
                ))
            )
            ready_helper = (
                None
                if helper is None
                or helper_rebind is not None
                and not retains_helper_sessions
                else self._ready_request_helper(ticket, helper)
            )
            helper_is_ready = bool(
                helper is not None
                and ready_helper is not None
                and (
                    attached
                    or self._attach_ready_request_helper(
                        ticket,
                        slot_id=slot_id,
                        token_index=token_index,
                        at_us=at_us,
                        fraction_ppm=(
                            0 if not isinstance(attachment, Mapping)
                            else int(attachment.get("fraction_ppm", 0))
                        ),
                        ready_helper=ready_helper,
                    )
                )
            )
            if helper is not None and helper_is_ready:
                adaptive_state = self._adaptive_decode.snapshot(request_id)
                identity = (
                    None if adaptive_state["helper_available"] else
                    self._adaptive_layout_identity_sha256(ticket, helper.phone_layout_generation)
                )
                self._record_late_helper_adoption(
                    request_id, helper, token_index=token_index, at_us=at_us,
                )
                self._adaptive_decode.helper_ready(
                    request_id,
                    phone_layout_generation=(
                        helper.phone_layout_generation
                    ),
                    phone_layout_geometry_sha256=(
                        helper.phone_layout_geometry_sha256
                    ),
                    candidates=(
                        late.candidates
                        if late is not None
                        and not adaptive_state["helper_available"]
                        else None
                    ),
                    component_capability_sha256=(
                        late.component.identity_sha256
                        if late is not None
                        and not adaptive_state["helper_available"]
                        else None
                    ),
                    ticket_policy=(
                        late.ticket_policy
                        if late is not None
                        and not adaptive_state["helper_available"]
                        else None
                    ),
                    helper_evidence_state=(
                        late.evidence_state
                        if late is not None else None
                    ),
                    ready_at_token_index=token_index,
                    allow_assumed_phone_power_for_operational_selection=(
                        self._adaptive_start_allow_assumed_phone_power(
                            ticket.execution_plan, helper, None
                        )
                        if not adaptive_state["helper_available"] else None
                    ),
                    **({} if identity is None else {"phone_layout_identity_sha256": identity}),
                )
                self._record_adaptive_helper_energy_policy_binding(
                    ticket, helper, adaptive_state, at_us,
                )
            elif helper is not None:
                self._adaptive_decode.helper_unavailable(request_id)
            directive = self._adaptive_decode.boundary(
                request_id,
                slot_id=slot_id,
                token_index=token_index,
                at_us=at_us,
                terminal=terminal,
            )
            return self._track_adaptive_directive(
                request_id,
                directive,
                at_us=at_us,
                token_index=token_index,
            )
        except AdaptiveDecodeError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    @_runtime_serialized
    def seal_adaptive_decode_tail(
        self,
        request_id: str,
        *,
        slot_id: int,
        token_index: int,
        reason: str,
    ) -> None:
        """Seal the measured decode windows before a server releases its slot."""
        try:
            self._adaptive_decode.seal_tail(
                request_id,
                slot_id=slot_id,
                token_index=token_index,
                reason=reason,
            )
        except AdaptiveDecodeError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    @_runtime_serialized
    def acknowledge_adaptive_decode_control(
        self,
        request_id: str,
        acknowledgement: AdaptiveDecodePolicyAck,
        transition_observation: AdaptiveDecodeRawWindowObservation | None = None,
    ) -> AdaptiveDecodeDirective:
        try:
            with self._transaction(errors=Exception, convert=False):
                return self._acknowledge_adaptive_decode_control_transaction(
                    request_id,
                    acknowledgement,
                    transition_observation,
                )
        except Exception as exc:
            if isinstance(exc, UnifiedScheduleError):
                raise
            if isinstance(
                exc, (AdaptiveDecodeError, ModelPlacementControllerError)
            ):
                raise UnifiedScheduleError(str(exc)) from exc
            raise

    def _acknowledge_adaptive_decode_control_transaction(
        self,
        request_id: str,
        acknowledgement: AdaptiveDecodePolicyAck,
        transition_observation: AdaptiveDecodeRawWindowObservation | None,
    ) -> AdaptiveDecodeDirective:
        ticket = self.runtime_execution_ticket(request_id)
        baseline, candidates, _ticket_policy = (
            self._adaptive_policies_from_ticket(ticket)
        )
        rebind = (
            self._model_placement_controller
            .request_helper_rebind_state(request_id)
        )
        self._report_helper_phone_disturbance(request_id)
        directive = self._adaptive_decode.acknowledge(
            request_id,
            acknowledgement,
            transition_observation=transition_observation,
        )
        drain_acknowledged = bool(
            rebind is not None
            and rebind.get("drain_policy_sha256")
                == acknowledgement.policy_hash
            and tuple(rebind.get("target_allowed_session_ids", ()))
        )
        if drain_acknowledged:
            assert rebind is not None
            self._model_placement_controller\
                .mark_request_helper_rebind_quiesced(
                    request_id,
                    int(rebind["target_generation"]),
                    observed_at_us=acknowledgement.applied_at_us,
                    allowed_session_ids=tuple(rebind[
                        "target_allowed_session_ids"
                    ]),
                    drain_policy_sha256=acknowledgement.policy_hash,
                )
        elif acknowledgement.policy_hash == baseline.policy_hash:
            self._release_request_helper_leases(
                request_id, acknowledgement.applied_at_us
            )
            binding = self._model_placement_controller.request_binding(
                request_id
            )
            attachment = (
                None if binding is None else
                binding.get("helper_attachment")
            )
            if rebind is not None:
                if (
                    binding is None
                    or int(binding.get("fraction_ppm", -1)) != 0
                    or not isinstance(attachment, Mapping)
                    or int(attachment.get("fraction_ppm", -1)) != 0
                    or tuple(attachment.get("lease_tokens", ()))
                ):
                    raise UnifiedScheduleError(
                        "acknowledged helper rebind is not at 0%"
                    )
                self._model_placement_controller\
                    .mark_request_helper_rebind_quiesced(
                        request_id,
                        int(rebind["target_generation"]),
                        observed_at_us=acknowledgement.applied_at_us,
                    )
        self._record_applied_helper_fraction(
            ticket,
            acknowledgement,
            (baseline, *candidates),
        )
        return self._track_adaptive_directive(
            request_id,
            directive,
            at_us=acknowledgement.applied_at_us,
            token_index=acknowledgement.applied_token_index,
        )

    def _record_assistance_decision(self, request_id, directive, token_index, at_us):
        snapshot = self._adaptive_decode.snapshot(request_id)
        policy = (directive.control.policy if directive.control is not None
                  else self._adaptive_decode.active_policy(request_id)
                  if directive.target_token_index is not None else None)
        if policy is None:
            return
        reason = directive.reason
        if policy.baseline and reason in {"WINDOW_OPENED", "POLICY_CHANGE_REQUIRED"}:
            if not snapshot["helper_available"]:
                reason = "PHONE_HELPER_UNAVAILABLE"
            elif snapshot["stage"] in {"cached_baseline", "verification_baseline", "verification_candidate"}:
                reason = "VERIFICATION"
            elif not snapshot["candidate_policy_hashes"]:
                reason = "NO_COMPATIBLE_HELPER"
            else:
                reason = snapshot["zero_assistance_reason"]
        keys = (
            "incumbent_policy_hash", "incumbent_fraction_ppm", "incumbent_context_sha256",
            "challenger_policy_hash", "challenger_fraction_ppm", "acknowledged_policy_hash",
            "context_identity_sha256", "remaining_output_tokens", "remaining_probe_tokens",
            "estimated_exploration_overhead_uj", "probe_budget", "probe_attempts", "verification",
            "helper_layout_generation", "helper_layout_geometry_sha256", "helper_evidence_state",
            "maximum_latency_ppm", "minimum_energy_saving_ppm", "evidence", "eliminated_policy_reasons",
            "context_monitor_prior", "pending_helper_refresh_policy_hash", "window_role",
            "execution_context_available", "observed_control_cost_us", "observed_control_tokens",
            "warmup_latency_us_by_policy", "external_activity_sha256", "server_policy",
            "helper_disturbance",
        )
        identity = snapshot["helper_layout_identity_sha256"]
        self._model_placement_controller.record_request_helper_event(
            request_id, "ASSISTANCE_DECISION", at_us,
            {"token_index": token_index, "target_token_index": directive.target_token_index,
             "reason": reason, "selected_fraction_ppm": policy.split_fraction_ppm,
             "policy_hash": policy.policy_hash, "phase": "INTENT",
             "incumbent_retained": policy.policy_hash == snapshot["incumbent_policy_hash"],
             **({} if "device_sets" not in snapshot else {
                 "active_batch": snapshot["active_batch"],
                 "device_set": list(policy_device_set(policy)),
                 "device_layer_masks": [list(row) for row in policy.device_layer_masks]}),
             **{key: snapshot[key] for key in keys},
             **({} if identity is None else {"helper_layout_identity_sha256": identity}),
             # batch_growth_verdict_inheritance: the smaller composition whose verdict runs here
             **({} if reason != INHERITED_REASON else {
                 "inherited_from_batch": (snapshot["server_policy"] or {}).get("inherited_from_batch")})},
        )

    def _record_applied_helper_fraction(
        self,
        ticket: RuntimeRequestTicket,
        acknowledgement: AdaptiveDecodePolicyAck,
        policies: tuple[AdaptiveDecodePolicy, ...],
    ) -> None:
        request = getattr(ticket, "request", None)
        model = getattr(ticket, "model", None)
        ticket_id = getattr(ticket, "ticket_id", None)
        if request is None or model is None or ticket_id is None:
            return
        matching = tuple(
            policy for policy in policies
            if policy.policy_hash == acknowledgement.policy_hash
        )
        binding = self._model_placement_controller.request_binding(
            request.request_id
        )
        envelope = (
            None if binding is None else binding.get("helper_envelope")
        )
        if len(matching) != 1 or not isinstance(envelope, Mapping):
            return
        identities = envelope.get("phone_session_identities", ())
        self._model_placement_controller.record_request_helper_event(
            request.request_id,
            "FRACTION_APPLIED",
            acknowledgement.applied_at_us,
            {
                "accepted": True,
                "artifact_sha256": model.artifact_sha256,
                "desktop_parent_placement_sha256": (
                    envelope.get("desktop_placement_sha256")
                ),
                "operator_plan_sha256": (
                    envelope.get("operator_plan_sha256")
                ),
                "phone_layout_generation": (
                    envelope.get("phone_layout_generation")
                ),
                "phone_layout_geometry_sha256": (
                    envelope.get("phone_layout_geometry_sha256")
                ),
                "request_ticket_id": ticket_id,
                "selected_fraction_ppm": matching[0].split_fraction_ppm,
                "session_generation_by_id": {
                    str(row["session_id"]): int(row["session_generation"])
                    for row in identities
                    if isinstance(row, Mapping)
                },
                "template_layout_generation": (
                    envelope.get("phone_layout_generation")
                ),
            },
        )

    def _compatible_helper_batch_change(self, request_id, boundary, next_batch):
        ticket = self.runtime_execution_ticket(request_id)
        helper = self._request_helper_envelope(ticket)
        if helper is None:
            return False
        if helper.operator_plan_sha256 != boundary.policy.operator_plan_sha256:
            helper = self._request_helper_envelope_history.get(request_id, {}).get(
                boundary.policy.operator_plan_sha256
            )
        return bool(
            helper is not None
            and helper.desktop_placement_sha256 == boundary.policy.desktop_placement_sha256
            and next_batch <= helper.helper_plan.execution_contract.maximum_batch_size
            and self._ready_request_helper(ticket, helper) is not None
        )

    def _report_helper_phone_disturbance(self, request_id: str) -> None:
        """A session loading on the phone that serves the request's helper disturbs its
        phone windows (the load shares the HTP and USB with them)."""
        binding = self._model_placement_controller.request_binding(request_id)
        envelope = None if binding is None else binding.get("helper_envelope")
        reason = None
        if isinstance(envelope, Mapping):
            own = set(envelope.get("phone_session_ids", ()))
            states = self._model_placement_controller.phone_session_states()
            phones = {_phone_device(row.endpoint) for row in states if row.session_id in own}
            if any(row.state == "LOADING" and _phone_device(row.endpoint) in phones
                   for row in states):
                reason = "HELPER_PHONE_SESSION_LOAD"
        self._adaptive_decode.helper_disturbance(request_id, reason=reason)

    def _external_desktop_activity(self, request_id, boundary):
        """Measurement context of a window: other work charged on server resources."""
        lookup = getattr(self, "runtime_external_desktop_activity", None)
        if lookup is None:
            return None
        try:
            ticket = self.runtime_execution_ticket(request_id)
        except UnifiedScheduleError:
            return None
        return lookup(ticket.ticket_id, boundary.started_at_us, boundary.finished_at_us)

    @_runtime_serialized
    def record_adaptive_decode_window(
        self,
        request_id: str,
        boundary: AdaptiveDecodeWindowBoundary,
        observation: AdaptiveDecodeRawWindowObservation,
    ) -> AdaptiveDecodeDirective:
        try:
            before = self._adaptive_decode.snapshot(request_id)
            previous_batch = before["active_batch"]
            next_batch = observation.next_active_batch or observation.active_batch or previous_batch
            changed = observation.membership_changed or next_batch != previous_batch
            compatible = bool(changed and self._compatible_helper_batch_change(
                request_id, boundary, next_batch
            ))
            if changed and not compatible and not boundary.policy.baseline:
                self._adaptive_decode.helper_unavailable(request_id)
            external = (
                None if observation.external_activity_sha256 is not None
                else self._external_desktop_activity(request_id, boundary)
            )
            if external is not None:
                observation = replace(
                    observation, external_activity_sha256=external["sha256"],
                )
            previous_external = before["external_activity_sha256"]
            external_changed = bool(
                observation.external_activity_sha256 is not None
                and previous_external is not None
                and observation.external_activity_sha256 != previous_external
            )
            self._report_helper_phone_disturbance(request_id)
            directive = self._adaptive_decode.record_window(
                request_id, boundary, observation, compatible_batch_change=compatible,
            )
            if external_changed:
                after = self._adaptive_decode.snapshot(request_id)
                self._model_placement_controller.record_request_helper_event(
                    request_id, "CONTEXT_CHANGED", boundary.finished_at_us,
                    {"token_index": boundary.token_end, "previous_active_batch": previous_batch,
                     "active_batch": next_batch, "measurement_eligible": False,
                     "execution_compatible": not changed or compatible,
                     "previous_external_activity_sha256": previous_external,
                     "external_activity_sha256": observation.external_activity_sha256,
                     "external_desktop_ticket_ids": (
                         [] if external is None else list(external["ticket_ids"])),
                     "context_monitor_prior": after["context_monitor_prior"],
                     "reason": "EXTERNAL_DESKTOP_ACTIVITY_CHANGED",
                     "probe_tokens": after["probe_tokens"]},
                )
            if changed:
                self._model_placement_controller.record_request_helper_event(
                    request_id, "CONTEXT_CHANGED", boundary.finished_at_us,
                    {"token_index": boundary.token_end, "previous_active_batch": previous_batch,
                     "active_batch": next_batch, "measurement_eligible": False,
                     "execution_compatible": compatible,
                     "context_monitor_prior": self._adaptive_decode.snapshot(request_id)["context_monitor_prior"],
                     "reason": "LIVE_MEMBERSHIP_CHANGED", "probe_tokens":
                         self._adaptive_decode.snapshot(request_id)["probe_tokens"]},
                )
            next_policy = (
                directive.control.policy
                if directive.control is not None else
                self._adaptive_decode.active_policy(request_id)
                if directive.target_token_index is not None else None
            )
            if (
                not boundary.policy.baseline
                and next_policy is not None
                and not next_policy.baseline
            ):
                binding = (
                    self._model_placement_controller.request_binding(
                        request_id
                    )
                )
                envelope = (
                    None if binding is None else
                    binding.get("helper_envelope")
                )
                rebind = self._model_placement_controller\
                    .request_helper_rebind_state(request_id)
                retained_drain = bool(
                    rebind is not None
                    and tuple(rebind.get("target_allowed_session_ids", ()))
                    and rebind.get("drain_policy_sha256") == next_policy.policy_hash
                )
                if isinstance(envelope, Mapping) and not retained_drain:
                    owner_id, _bid = self._select_helper_window_owner(
                        request_id,
                        tuple(envelope.get("resource_ids", ())),
                        requested_fraction_ppm=next_policy.split_fraction_ppm,
                        observed_at_us=boundary.finished_at_us,
                    )
                    if owner_id != request_id:
                        if directive.control is not None:
                            directive = self._adaptive_decode.defer_control(
                                request_id,
                                directive.control,
                                "HELPER_WINDOW_NOT_SELECTED",
                                at_us=boundary.finished_at_us,
                            )
                        active_policy = self._adaptive_decode.active_policy(
                            request_id
                        )
                        if active_policy is not None and not active_policy.baseline:
                            directive = self._adaptive_decode.yield_helper_window(
                                request_id,
                                token_index=boundary.token_end,
                                at_us=boundary.finished_at_us,
                            )
            if (
                observation.completed_phone_calls is not None
                and observation.completed_phone_calls > 0
            ):
                self._model_placement_controller.record_request_helper_work(
                    request_id,
                    observation.completed_phone_calls,
                    observed_at_us=boundary.finished_at_us,
                )
            adaptive_state = self._adaptive_decode.snapshot(request_id)
            if (
                adaptive_state["helper_evidence_state"] == "LEARNING"
                and not boundary.policy.baseline
            ):
                self._model_placement_controller.record_request_helper_event(
                    request_id,
                    "LEARNING_WINDOW_RECORDED",
                    boundary.finished_at_us,
                    {
                        "completed_phone_calls": (
                            observation.completed_phone_calls
                        ),
                        "energy_attribution_kind": (
                            observation.energy_attribution_kind
                        ),
                        "energy_boundary_id": (
                            observation.energy_boundary_id
                        ),
                        "evidence_ids": list(observation.evidence_ids),
                        "evidence_state": "LEARNING",
                        "fleet_energy_uj": sum(
                            observation.fleet_energy_uj_by_domain.values()
                        ),
                        "latency_per_token_us": max(
                            1,
                            (
                                boundary.finished_at_us
                                - boundary.started_at_us
                            ) // boundary.token_count,
                        ),
                        "output_valid": observation.output_valid,
                        "qualification_state": "DIAGNOSTIC",
                        "split_fraction_ppm": (
                            boundary.policy.split_fraction_ppm
                        ),
                        # several helper phones: the device set and batch composition measured
                        **({} if not boundary.policy.device_layer_masks else {
                            "active_batch": adaptive_state["active_batch"],
                            "device_set": list(policy_device_set(boundary.policy)),
                            "token_count": boundary.token_count,
                        }),
                    },
                )
            return self._track_adaptive_directive(
                request_id,
                directive,
                at_us=boundary.finished_at_us,
                token_index=boundary.token_end,
            )
        except (
            AdaptiveDecodeError,
            ModelPlacementControllerError,
        ) as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    @_runtime_serialized
    def fail_adaptive_decode_control(
        self,
        request_id: str,
        control: AdaptiveDecodeControl,
        reason: str,
        *,
        at_us: int,
    ) -> AdaptiveDecodeDirective:
        try:
            self._model_placement_controller.record_request_helper_event(
                request_id,
                "CONTROL_FAILED",
                at_us,
                {
                    "plan_generation": control.plan_generation,
                    "policy_hash": control.policy.policy_hash,
                    "reason": reason,
                    "requested_fraction_ppm": (
                        control.policy.split_fraction_ppm
                    ),
                    "slot_id": control.slot_id,
                },
            )
            directive = self._adaptive_decode.control_failed(
                request_id, control, reason, at_us=at_us
            )
            snapshot = self._adaptive_decode.snapshot(request_id)
            token_index = snapshot.get("transition_start_token")
            return self._track_adaptive_directive(
                request_id,
                directive,
                at_us=at_us,
                token_index=(
                    0 if type(token_index) is not int else token_index
                ),
            )
        except AdaptiveDecodeError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    @_runtime_serialized
    def discard_stale_adaptive_decode_window(
        self,
        request_id: str,
        boundary: AdaptiveDecodeWindowBoundary,
        observation: AdaptiveDecodeRawWindowObservation,
        reason: str,
        *,
        terminal_token_index: int | None = None,
        terminal_at_us: int | None = None,
    ) -> AdaptiveDecodeDirective:
        try:
            return self._adaptive_decode.discard_stale_window(
                request_id, boundary, observation, reason,
                terminal_token_index=terminal_token_index,
                terminal_at_us=terminal_at_us,
            )
        except AdaptiveDecodeError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    @_runtime_serialized
    def discard_stale_adaptive_decode_control(
        self,
        request_id: str,
        control: AdaptiveDecodeControl,
        reason: str,
        *,
        at_us: int,
    ) -> AdaptiveDecodeDirective:
        try:
            directive = self._adaptive_decode.discard_stale_control(
                request_id, control, reason, at_us=at_us
            )
            return self._track_adaptive_directive(
                request_id,
                directive,
                at_us=at_us,
                token_index=0,
            )
        except AdaptiveDecodeError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    @_runtime_serialized
    def complete_adaptive_decode(
        self, request_id: str, terminal_status: str = "COMPLETED"
    ) -> AdaptiveDecodeGroupedObservation:
        try:
            result = self._adaptive_decode.complete(
                request_id, terminal_status
            )
            release_at_us = max(
                (row.finished_at_us for row in result.windows),
                default=0,
            )
            self._release_request_helper_leases(
                request_id, release_at_us
            )
            binding = self._model_placement_controller.request_binding(
                request_id
            )
            if binding is not None and binding.get(
                "helper_attachment"
            ) is not None:
                self._model_placement_controller.detach_request_helper(
                    request_id,
                    fallback_outcome="REQUEST_COMPLETED",
                    observed_at_us=release_at_us,
                )
                note_resource_release(
                    self, request_id, release_at_us, "DECODE_COMPLETED"
                )
            return result
        except AdaptiveDecodeError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    def preview_adaptive_decode_completion(
        self, request_id: str
    ) -> AdaptiveDecodeGroupedObservation:
        """Return the terminal proof before the physical commit succeeds."""
        try:
            return self._adaptive_decode.preview_completion(request_id)
        except AdaptiveDecodeError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    def adaptive_decode_snapshot(
        self, request_id: str
    ) -> Mapping[str, object]:
        try:
            return self._adaptive_decode.snapshot(request_id)
        except AdaptiveDecodeError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    def adaptive_decode_timing(
        self, request_id: str
    ) -> Mapping[str, int]:
        try:
            return self._adaptive_decode.timing(request_id)
        except AdaptiveDecodeError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    def adaptive_decode_grouped_observation(
        self, request_id: str
    ) -> AdaptiveDecodeGroupedObservation:
        try:
            return self._adaptive_decode.grouped_observation(request_id)
        except AdaptiveDecodeError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    def adaptive_decode_observation_state(self) -> Mapping[str, int]:
        return self._adaptive_decode.observation_state()

    def adaptive_decode_observation_snapshot(
        self,
    ) -> Mapping[str, object]:
        return self._adaptive_decode.observation_snapshot()

    @_runtime_serialized
    def load_adaptive_decode_observations(
        self,
        value: object,
        *,
        source_catalog: RuntimeCapabilityCatalog | None = None,
    ) -> None:
        try:
            self._adaptive_decode.load_observations(value)
            if source_catalog is not None:
                if not isinstance(source_catalog, RuntimeCapabilityCatalog):
                    raise AdaptiveDecodeError(
                        "adaptive observation source catalog is invalid"
                    )
                groups = value.get("groups") if type(value) is dict else None
                if type(groups) is not list:
                    raise AdaptiveDecodeError(
                        "adaptive observation source groups are invalid"
                    )
                profiles = frozenset(
                    row.get("planning_profile_sha256")
                    for row in groups
                    if type(row) is dict
                )
                if not profiles or any(
                    type(profile) is not str for profile in profiles
                ):
                    raise AdaptiveDecodeError(
                        "adaptive observation source profile is invalid"
                    )
                self._adaptive_observation_sources[
                    canonical_sha256(source_catalog)
                ] = (source_catalog, profiles)
            for manifest in self._runtime_manifests.values():
                self._model_placement_controller.notify(
                    manifest.artifact_sha256,
                    "LEARNING_GENERATION_CHANGED",
                    0,
                )
        except AdaptiveDecodeError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    @_runtime_serialized
    def merge_adaptive_decode_observations(self, value: object) -> None:
        try:
            self._adaptive_decode.load_observations(value, merge=True)
            for manifest in self._runtime_manifests.values():
                self._model_placement_controller.notify(
                    manifest.artifact_sha256,
                    "LEARNING_GENERATION_CHANGED",
                    0,
                )
        except AdaptiveDecodeError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
