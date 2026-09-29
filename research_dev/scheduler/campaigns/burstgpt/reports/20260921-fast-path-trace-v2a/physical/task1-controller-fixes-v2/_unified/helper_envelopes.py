"""Request helper envelopes: opportunity policies, verified ready helpers, rematerialization.

Mixin of ``UnifiedScheduler``; methods were moved here verbatim and rely on the
state initialised in ``UnifiedScheduler.__init__``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from .._internal.runtime_cost import RuntimeExecutorBinding
from .._internal.runtime_capabilities import (
    HeterogeneousRuntimeSnapshot,
    RuntimeCompositeExecutorCapability,
    RuntimeExecutorCapability,
    RuntimeExecutorState,
)
from .._internal.model_placement_controller import (
    ModelPhoneResidencyLayout,
    RequestHelperEnvelopeBinding,
)
from .._internal.phone_shards import PhoneFfnResidencyLayout
from .._internal.runtime_plan import (
    AutomatedCandidateSet,
    AutomatedRouteCandidate,
    HelperOpportunity,
    PhoneSessionReplacementAuthorization,
    RuntimeHelperExecutionEnvelope,
    RuntimeExecutionPlan,
)
from .._internal.runtime_residency_cohorts import RuntimeResidencyComponentIdentity
from .._internal.runtime_controller import RuntimeRequestTicket
from .._internal.adaptive_decode_contracts import AdaptiveDecodePolicy
from .common import _RequestHelperPreparation, _LateRequestHelperContext, _runtime_serialized

from .helper_envelopes_ops.common import (
    _ReadyHelperMaterialization as _ReadyHelperMaterialization,
    _ReadyHelperParentUnavailable as _ReadyHelperParentUnavailable,
)
from .helper_envelopes_ops import cleanup as _cleanup
from .helper_envelopes_ops import late_attachment as _late_attachment
from .helper_envelopes_ops import materialization as _materialization
from .helper_envelopes_ops import policies as _policies
from .helper_envelopes_ops import ready_plan as _ready_plan
from .helper_envelopes_ops import refresh as _refresh
from .helper_envelopes_ops import replacement as _replacement
from .helper_envelopes_ops import templates as _templates

__all__ = [
    'AdaptiveDecodePolicy',
    'AutomatedCandidateSet',
    'AutomatedRouteCandidate',
    'HelperEnvelopeMixin',
    'HelperOpportunity',
    'HeterogeneousRuntimeSnapshot',
    'ModelPhoneResidencyLayout',
    'PhoneFfnResidencyLayout',
    'PhoneSessionReplacementAuthorization',
    'RequestHelperEnvelopeBinding',
    'RuntimeCompositeExecutorCapability',
    'RuntimeExecutionPlan',
    'RuntimeExecutorBinding',
    'RuntimeExecutorCapability',
    'RuntimeExecutorState',
    'RuntimeHelperExecutionEnvelope',
    'RuntimeRequestTicket',
    'RuntimeResidencyComponentIdentity',
    '_LateRequestHelperContext',
    '_ReadyHelperMaterialization',
    '_ReadyHelperParentUnavailable',
    '_RequestHelperPreparation',
    '_cleanup',
    '_late_attachment',
    '_materialization',
    '_policies',
    '_ready_plan',
    '_refresh',
    '_replacement',
    '_runtime_serialized',
    '_templates',
]


class HelperEnvelopeMixin:
    """Request helper envelopes: opportunity policies, verified ready helpers, rematerialization."""

    @staticmethod
    def _reusable_phone_helper_template(
        helper: RuntimeHelperExecutionEnvelope,
    ) -> RuntimeHelperExecutionEnvelope:
        """Remove preparation-transaction state from a reusable helper."""
        return _templates._reusable_phone_helper_template(helper)

    @staticmethod
    def _ready_helper_event_values(
        ticket: RuntimeRequestTicket,
        layout: ModelPhoneResidencyLayout,
        *,
        helper: RuntimeHelperExecutionEnvelope | None = None,
        selected_fraction_ppm: int = 0,
        accepted: bool,
        reason: str | None = None,
        remaining_opportunity_tokens: int | None = None,
    ) -> dict[str, object]:
        return _templates._ready_helper_event_values(
            ticket,
            layout,
            helper=helper,
            selected_fraction_ppm=selected_fraction_ppm,
            accepted=accepted,
            reason=reason,
            remaining_opportunity_tokens=remaining_opportunity_tokens,
        )

    def _record_ready_helper_event_once(
        self,
        ticket: RuntimeRequestTicket,
        layout: ModelPhoneResidencyLayout,
        kind: str,
        observed_at_us: int,
        *,
        helper: RuntimeHelperExecutionEnvelope | None = None,
        selected_fraction_ppm: int = 0,
        accepted: bool,
        reason: str | None = None,
        remaining_opportunity_tokens: int | None = None,
    ) -> None:
        return _templates._record_ready_helper_event_once(
            self,
            ticket,
            layout,
            kind,
            observed_at_us,
            helper=helper,
            selected_fraction_ppm=selected_fraction_ppm,
            accepted=accepted,
            reason=reason,
            remaining_opportunity_tokens=remaining_opportunity_tokens,
        )

    def _authoritative_ready_helper_template(
        self,
        *,
        artifact_sha256: str,
        desktop_parent_route_id: str,
        desktop_placement_sha256: str,
        baseline_executor_id: str,
        allow_parent_route_rebind: bool = False,
    ) -> RuntimeHelperExecutionEnvelope | None:
        """Return the exact system-owned helper for the READY session view."""
        return _templates._authoritative_ready_helper_template(
            self,
            artifact_sha256=artifact_sha256,
            desktop_parent_route_id=desktop_parent_route_id,
            desktop_placement_sha256=desktop_placement_sha256,
            baseline_executor_id=baseline_executor_id,
            allow_parent_route_rebind=allow_parent_route_rebind,
        )

    def _phone_helper_authorization_layout(
        self,
        layout: ModelPhoneResidencyLayout,
    ) -> PhoneFfnResidencyLayout:
        """Bind a delayed proposal to its exact authoritative source."""
        return _templates._phone_helper_authorization_layout(self, layout)

    def _build_helper_envelope(
        self,
        *,
        artifact_sha256: str,
        desktop_parent_route_id: str,
        desktop_placement_sha256: str,
        helper_plan: RuntimeExecutionPlan,
        helper_binding: RuntimeExecutorBinding,
        layout: ModelPhoneResidencyLayout,
    ) -> RuntimeHelperExecutionEnvelope:
        return _policies._build_helper_envelope(
            self,
            artifact_sha256=artifact_sha256,
            desktop_parent_route_id=desktop_parent_route_id,
            desktop_placement_sha256=desktop_placement_sha256,
            helper_plan=helper_plan,
            helper_binding=helper_binding,
            layout=layout,
        )

    @staticmethod
    def _request_helper_envelope_binding(
        helper: RuntimeHelperExecutionEnvelope,
        layout: ModelPhoneResidencyLayout,
    ) -> RequestHelperEnvelopeBinding:
        return _policies._request_helper_envelope_binding(helper, layout)

    @staticmethod
    def _ticket_adaptive_policy_records(
        ticket: RuntimeRequestTicket,
    ) -> tuple[AdaptiveDecodePolicy, tuple[AdaptiveDecodePolicy, ...]]:
        return _policies._ticket_adaptive_policy_records(ticket)

    def _helper_opportunity_policies(
        self,
        ticket: RuntimeRequestTicket,
        opportunity: HelperOpportunity,
    ) -> tuple[
        AdaptiveDecodePolicy,
        tuple[AdaptiveDecodePolicy, ...],
        AdaptiveDecodePolicy,
    ] | None:
        return _policies._helper_opportunity_policies(self, ticket, opportunity)

    @staticmethod
    def _candidate_set_for_ready_phone_layout(
        candidate_set: AutomatedCandidateSet,
        artifact_sha256: str,
        layout: ModelPhoneResidencyLayout,
        *, usb_batch_plan: str | None = None,
    ) -> AutomatedCandidateSet:
        return _policies._candidate_set_for_ready_phone_layout(
            candidate_set,
            artifact_sha256,
            layout,
            usb_batch_plan=usb_batch_plan,
        )

    def _authorize_phone_helper_plan(
        self,
        plan: RuntimeExecutionPlan,
        binding: RuntimeExecutorBinding,
        layout: ModelPhoneResidencyLayout,
        *,
        model_id: str,
        artifact_sha256: str,
    ) -> tuple[RuntimeExecutionPlan, RuntimeExecutorBinding]:
        return _policies._authorize_phone_helper_plan(
            self,
            plan,
            binding,
            layout,
            model_id=model_id,
            artifact_sha256=artifact_sha256,
        )

    @staticmethod
    def _snapshot_with_owned_base_executor(
        ticket: RuntimeRequestTicket,
        snapshot: HeterogeneousRuntimeSnapshot,
    ) -> HeterogeneousRuntimeSnapshot:
        """Expose the acquired request's own base slot for rematerialization."""
        return _ready_plan._snapshot_with_owned_base_executor(ticket, snapshot)

    def _resolve_verified_ready_helper_plan(
        self,
        ticket: RuntimeRequestTicket,
        layout: ModelPhoneResidencyLayout,
    ) -> tuple[RuntimeExecutionPlan, RuntimeExecutorBinding] | None:
        return _ready_plan._resolve_verified_ready_helper_plan(self, ticket, layout)

    def _ready_parent_helper_coordinator(
        self,
        ticket: RuntimeRequestTicket,
        layout: ModelPhoneResidencyLayout,
        snapshot: HeterogeneousRuntimeSnapshot,
    ) -> RuntimeCompositeExecutorCapability:
        return _ready_plan._ready_parent_helper_coordinator(self, ticket, layout, snapshot)

    @contextmanager
    def _ready_layout_compiler_view(
        self,
        layout: ModelPhoneResidencyLayout,
    ) -> Iterator[None]:
        """Generate a late helper from the authoritative READY layout."""
        yield from _ready_plan._ready_layout_compiler_view(self, layout)

    def _verified_helper_endpoint_states(
        self,
        ticket: RuntimeRequestTicket,
        layout: ModelPhoneResidencyLayout,
        snapshot: HeterogeneousRuntimeSnapshot,
        helper_plan: RuntimeExecutionPlan,
        helper_binding: RuntimeExecutorBinding,
    ) -> tuple[object, RuntimeExecutorState, object, RuntimeExecutorState, str]:
        return _ready_plan._verified_helper_endpoint_states(
            self,
            ticket,
            layout,
            snapshot,
            helper_plan,
            helper_binding,
        )

    def _verified_helper_coordinator_states(
        self,
        ticket: RuntimeRequestTicket,
        layout: ModelPhoneResidencyLayout,
        snapshot: HeterogeneousRuntimeSnapshot,
        coordinator: RuntimeCompositeExecutorCapability,
    ) -> tuple[
        RuntimeCompositeExecutorCapability, RuntimeExecutorState,
        RuntimeExecutorCapability, RuntimeExecutorState, str,
    ]:
        return _ready_plan._verified_helper_coordinator_states(
            self,
            ticket,
            layout,
            snapshot,
            coordinator,
        )

    def _effective_verified_phone_state(
        self,
        phone_capability: object,
        phone_state: RuntimeExecutorState,
        phone_device_id: str,
        phone_safety_state: RuntimeExecutorState | None,
    ) -> RuntimeExecutorState:
        return _ready_plan._effective_verified_phone_state(
            self,
            phone_capability,
            phone_state,
            phone_device_id,
            phone_safety_state,
        )

    def _snapshot_with_verified_ready_helper(
        self,
        ticket: RuntimeRequestTicket,
        layout: ModelPhoneResidencyLayout,
        snapshot: HeterogeneousRuntimeSnapshot,
        phone_safety_state: RuntimeExecutorState | None,
    ) -> HeterogeneousRuntimeSnapshot:
        """Expose only the exact helper proven by a READY layout."""
        return _ready_plan._snapshot_with_verified_ready_helper(
            self,
            ticket,
            layout,
            snapshot,
            phone_safety_state,
        )

    @staticmethod
    def _ready_helper_parent_rejection_reasons(
        ticket: RuntimeRequestTicket,
        generated_baseline: AutomatedRouteCandidate,
        opportunity: HelperOpportunity,
    ) -> tuple[str, ...]:
        """Validate a regenerated helper against the acquired desktop base."""
        return _ready_plan._ready_helper_parent_rejection_reasons(
            ticket,
            generated_baseline,
            opportunity,
        )

    @staticmethod
    def _helper_changed_session_ids(
        layout: ModelPhoneResidencyLayout,
        helper_plan: RuntimeExecutionPlan,
    ) -> tuple[str, ...]:
        """Scope a layout's changed sessions to one helper's own shards."""
        return _replacement._helper_changed_session_ids(layout, helper_plan)

    def _phone_session_replacement_authorization(
        self,
        target: ModelPhoneResidencyLayout,
    ) -> PhoneSessionReplacementAuthorization | None:
        """Bind a partial target to the exact selected source session."""
        return _replacement._phone_session_replacement_authorization(self, target)

    def _validate_phone_session_replacement_authorization(
        self,
        target: ModelPhoneResidencyLayout,
        authorization: PhoneSessionReplacementAuthorization | None,
    ) -> None:
        """Validate the selected session without changing its assignment."""
        return _replacement._validate_phone_session_replacement_authorization(
            self,
            target,
            authorization,
        )

    def _prevalidate_target_layout_helper_envelopes(
        self,
        target: ModelPhoneResidencyLayout,
        *,
        request_id: str,
        observed_at_us: int,
    ) -> None:
        """Check every retained and new helper's session scope before a load.

        Each acquired request covered by the target layout will receive a
        helper envelope whose changed sessions are scoped to its own shards.
        A retained helper must not see its attached sessions as changed, and
        a helper whose sessions are replaced must already be quiescing.
        """
        return _replacement._prevalidate_target_layout_helper_envelopes(
            self,
            target,
            request_id=request_id,
            observed_at_us=observed_at_us,
        )

    def _materialize_phone_layout_preparation_envelope(
        self,
        ticket: RuntimeRequestTicket,
        layout: ModelPhoneResidencyLayout,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
    ) -> RuntimeHelperExecutionEnvelope | None:
        """Materialize one exact phone-only transition for a target layout."""
        return _replacement._materialize_phone_layout_preparation_envelope(
            self,
            ticket,
            layout,
            snapshot,
            observed_at_us,
        )

    def _record_preparation_envelope_materialized(
        self,
        ticket: RuntimeRequestTicket,
        layout: ModelPhoneResidencyLayout,
        helper: RuntimeHelperExecutionEnvelope,
        opportunity: HelperOpportunity,
        observed_at_us: int,
    ) -> None:
        return _replacement._record_preparation_envelope_materialized(
            self,
            ticket,
            layout,
            helper,
            opportunity,
            observed_at_us,
        )

    @_runtime_serialized
    def runtime_request_helper_preparation_envelope(
        self,
        request_id: str,
        *,
        expected_ticket_id: str,
        observed_at_us: int,
        snapshot: HeterogeneousRuntimeSnapshot,
    ) -> RuntimeHelperExecutionEnvelope | None:
        """Return an exact scheduler-issued helper transition envelope."""
        return _replacement.runtime_request_helper_preparation_envelope(
            self,
            request_id,
            expected_ticket_id=expected_ticket_id,
            observed_at_us=observed_at_us,
            snapshot=snapshot,
        )

    @_runtime_serialized
    def runtime_background_helper_preparation_allowed(
        self,
        request_id: str,
        *,
        expected_ticket_id: str,
    ) -> bool:
        """Permit demanded phone preparation for helper-eligible requests."""
        return _replacement.runtime_background_helper_preparation_allowed(
            self,
            request_id,
            expected_ticket_id=expected_ticket_id,
        )

    def _ready_helper_candidate_records(
        self,
        ticket: RuntimeRequestTicket,
        layout: ModelPhoneResidencyLayout,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
        phone_safety_state: RuntimeExecutorState | None,
    ) -> tuple[
        object,
        AutomatedCandidateSet,
        HelperOpportunity,
        object,
        str,
        AdaptiveDecodePolicy,
    ]:
        return _materialization._ready_helper_candidate_records(
            self,
            ticket,
            layout,
            snapshot,
            observed_at_us,
            phone_safety_state,
        )

    def _build_ready_helper_materialization(
        self,
        ticket: RuntimeRequestTicket,
        layout: ModelPhoneResidencyLayout,
        manifest: object,
        candidate_set: AutomatedCandidateSet,
        opportunity: HelperOpportunity,
        assisted_candidate: object,
        generated_parent_route_id: str,
        current_baseline: AdaptiveDecodePolicy,
    ) -> _ReadyHelperMaterialization:
        return _materialization._build_ready_helper_materialization(
            self,
            ticket,
            layout,
            manifest,
            candidate_set,
            opportunity,
            assisted_candidate,
            generated_parent_route_id,
            current_baseline,
        )

    def _bind_ready_helper_materialization(
        self,
        ticket: RuntimeRequestTicket,
        layout: ModelPhoneResidencyLayout,
        materialized: _ReadyHelperMaterialization,
        observed_at_us: int,
    ) -> bool:
        return _materialization._bind_ready_helper_materialization(
            self,
            ticket,
            layout,
            materialized,
            observed_at_us,
        )

    def _publish_ready_helper_materialization(
        self,
        ticket: RuntimeRequestTicket,
        layout: ModelPhoneResidencyLayout,
        materialized: _ReadyHelperMaterialization,
        masked_rebind: bool,
        observed_at_us: int,
    ) -> None:
        return _materialization._publish_ready_helper_materialization(
            self,
            ticket,
            layout,
            materialized,
            masked_rebind,
            observed_at_us,
        )

    def _rematerialize_ready_layout_helper(
        self,
        ticket: RuntimeRequestTicket,
        layout: ModelPhoneResidencyLayout,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
        phone_safety_state: RuntimeExecutorState | None,
    ) -> None:
        return _materialization._rematerialize_ready_layout_helper(
            self,
            ticket,
            layout,
            snapshot,
            observed_at_us,
            phone_safety_state,
        )

    def _retain_usable_ready_layout_helper(
        self,
        ticket: RuntimeRequestTicket,
        layout: ModelPhoneResidencyLayout,
        observed_at_us: int,
    ) -> bool:
        """Keep an exact session helper when another session is published."""
        return _materialization._retain_usable_ready_layout_helper(
            self,
            ticket,
            layout,
            observed_at_us,
        )

    def _handle_ready_helper_materialization_error(
        self,
        ticket: RuntimeRequestTicket,
        layout: ModelPhoneResidencyLayout,
        error: Exception,
        observed_at_us: int,
    ) -> None:
        return _materialization._handle_ready_helper_materialization_error(
            self,
            ticket,
            layout,
            error,
            observed_at_us,
        )

    def _ready_helper_configuration(
        self, ticket: RuntimeRequestTicket, layout: ModelPhoneResidencyLayout,
    ) -> str:
        return _refresh._ready_helper_configuration(self, ticket, layout)

    def _ready_helper_parent_rejection_unchanged(
        self, ticket: RuntimeRequestTicket, layout: ModelPhoneResidencyLayout,
    ) -> bool:
        return _refresh._ready_helper_parent_rejection_unchanged(self, ticket, layout)

    def _rematerialize_ready_layout_helpers(
        self,
        layout: ModelPhoneResidencyLayout,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
        phone_safety_state: RuntimeExecutorState | None = None,
    ) -> tuple[str, ...]:
        """Bind exact helper plans once after one layout becomes READY."""
        return _refresh._rematerialize_ready_layout_helpers(
            self,
            layout,
            snapshot,
            observed_at_us,
            phone_safety_state,
        )

    def _ready_layout_phone_safety_state(
        self,
        layout: ModelPhoneResidencyLayout,
    ) -> RuntimeExecutorState | None:
        return _refresh._ready_layout_phone_safety_state(self, layout)

    @_runtime_serialized
    def runtime_ready_helper_refresh_needed(
        self,
        request_id: str,
        *,
        expected_ticket_id: str,
    ) -> bool:
        return _refresh.runtime_ready_helper_refresh_needed(
            self,
            request_id,
            expected_ticket_id=expected_ticket_id,
        )

    @_runtime_serialized
    def refresh_ready_request_helper(
        self,
        request_id: str,
        *,
        expected_ticket_id: str,
        observed_at_us: int,
        snapshot: HeterogeneousRuntimeSnapshot,
    ) -> bool:
        """Materialize the current READY helper for one acquired request."""
        return _refresh.refresh_ready_request_helper(
            self,
            request_id,
            expected_ticket_id=expected_ticket_id,
            observed_at_us=observed_at_us,
            snapshot=snapshot,
        )

    def _validated_cached_late_request_helper(
        self,
        request_id: str,
        cached: _LateRequestHelperContext,
    ) -> _LateRequestHelperContext | None:
        return _late_attachment._validated_cached_late_request_helper(self, request_id, cached)

    def _materialize_late_request_helper(
        self,
        ticket: RuntimeRequestTicket,
    ) -> _LateRequestHelperContext | None:
        return _late_attachment._materialize_late_request_helper(self, ticket)

    def _store_late_request_helper(
        self,
        request_id: str,
        helper: RuntimeHelperExecutionEnvelope,
        baseline_policy: AdaptiveDecodePolicy,
        policies: tuple[AdaptiveDecodePolicy, ...],
        ticket_policy: AdaptiveDecodePolicy,
        component: RuntimeResidencyComponentIdentity,
        evidence_state: str,
    ) -> _LateRequestHelperContext:
        return _late_attachment._store_late_request_helper(
            self,
            request_id,
            helper,
            baseline_policy,
            policies,
            ticket_policy,
            component,
            evidence_state,
        )

    def _remember_request_helper_envelope(
        self,
        request_id: str,
        helper: RuntimeHelperExecutionEnvelope,
    ) -> None:
        return _late_attachment._remember_request_helper_envelope(self, request_id, helper)

    @staticmethod
    def _phone_helper_endpoint_identity(
        helper: RuntimeHelperExecutionEnvelope,
    ) -> tuple[object, ...]:
        return _late_attachment._phone_helper_endpoint_identity(helper)

    @staticmethod
    def _phone_helper_endpoint_template_key(
        helper: RuntimeHelperExecutionEnvelope,
    ) -> tuple[object, ...]:
        return _late_attachment._phone_helper_endpoint_template_key(HelperEnvelopeMixin, helper)

    def _remember_phone_helper_endpoint_template(
        self,
        helper: RuntimeHelperExecutionEnvelope,
    ) -> None:
        return _late_attachment._remember_phone_helper_endpoint_template(self, helper)

    def _request_helper_envelope(
        self,
        ticket: RuntimeRequestTicket,
    ) -> RuntimeHelperExecutionEnvelope | None:
        return _late_attachment._request_helper_envelope(self, ticket)

    def _ready_request_helper(
        self,
        ticket: RuntimeRequestTicket,
        helper: RuntimeHelperExecutionEnvelope | None = None,
    ) -> tuple[
        RuntimeHelperExecutionEnvelope,
        ModelPhoneResidencyLayout,
        RuntimeResidencyComponentIdentity,
    ] | None:
        return _late_attachment._ready_request_helper(self, ticket, helper)

    @staticmethod
    def _helper_preparation_json(
        preparation: _RequestHelperPreparation,
    ) -> dict[str, object]:
        return _cleanup._helper_preparation_json(preparation)

    def _release_request_helper_preparation(
        self,
        preparation: _RequestHelperPreparation,
        at_us: int,
    ) -> None:
        return _cleanup._release_request_helper_preparation(self, preparation, at_us)

    def _cancel_request_helper_preparation(
        self,
        preparation: _RequestHelperPreparation,
        at_us: int,
    ) -> None:
        return _cleanup._cancel_request_helper_preparation(self, preparation, at_us)
