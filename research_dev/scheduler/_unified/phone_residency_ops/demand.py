"""PhoneResidencyMixin demand operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Mapping, Sequence

from ..._internal.policy import Request
from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.model_manifest import ModelManifest
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.phone_shards import PhoneFfnResidencyDemand
from ..._internal.offline_phone_residency import offline_phone_workload_sha256
from ..._internal.runtime_plan import AutomatedCandidateSet
from ..._internal.runtime_controller import RuntimeRequestTicket
from .common import _OfflineLearningDemand, _PhoneDemandDiscovery, _PhoneQueueDemand
from .reprovision import queued_demand_learning_rejections

_PHONE_POLICY_EVENT_KINDS = frozenset({"ASSISTANCE_DECISION", "FRACTION_APPLIED"})


def _offline_planning_request(
    request: Request,
    *,
    request_id: str,
    observed_at_us: int,
) -> Request:
    return replace(
        request,
        request_id=request_id,
        arrival_us=observed_at_us,
        deadline_us=max(
            request.deadline_us,
            observed_at_us + 86_400_000_000,
        ),
    )


def _offline_phone_requests(
    controller,
    requests_by_model: Mapping[str, Sequence[Request]],
) -> tuple[
    str,
    dict[str, tuple[str, Request]],
    dict[str, int],
]:
    try:
        workload_sha256 = offline_phone_workload_sha256(
            requests_by_model
        )
    except ValueError as exc:
        raise UnifiedScheduleError(str(exc)) from exc
    request_by_artifact = {}
    queued_work_by_artifact = {}
    for model_id, raw_requests in sorted(requests_by_model.items()):
        manifest = controller.runtime_model_manifest(model_id)
        requests = tuple(raw_requests)
        representative = max(
            requests,
            key=lambda row: (
                row.output_tokens,
                row.input_tokens,
                row.request_id,
            ),
        )
        if manifest.artifact_sha256 in request_by_artifact:
            raise UnifiedScheduleError(
                "offline phone artifact has multiple model identities"
            )
        request_by_artifact[manifest.artifact_sha256] = (
            model_id, representative
        )
        queued_work_by_artifact[manifest.artifact_sha256] = sum(
            row.output_tokens for row in requests
        )
    return (
        workload_sha256,
        request_by_artifact,
        queued_work_by_artifact,
    )


def _offline_phone_discovery(
    controller,
    request_by_artifact: Mapping[str, tuple[str, Request]],
    queued_work_by_artifact: Mapping[str, int],
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
) -> _PhoneDemandDiscovery:
    learning_by_artifact = {}
    for artifact_sha256, (model_id, request) in sorted(
        request_by_artifact.items()
    ):
        planning_request = controller._offline_planning_request(
            request,
            request_id=(
                "offline-residency-evidence-"
                + artifact_sha256[7:23]
            ),
            observed_at_us=observed_at_us,
        )
        candidate_set = controller._generate_automated_candidate_set(
            planning_request,
            controller.runtime_model_manifest(model_id),
            snapshot,
            observed_at_us,
            use_residency_holds=False,
            update_phone_residency_portfolio=False,
        )
        learning = controller._learning_phone_demand(
            candidate_set,
            planning_request,
            controller.runtime_model_manifest(model_id),
            queued_work_by_artifact[artifact_sha256],
        )
        if learning is not None:
            learning_by_artifact[artifact_sha256] = learning
    discovered = controller._discover_phone_residency_demand(
        controller._automated_compiler(), queued_work_by_artifact
    )
    if not learning_by_artifact or (
        discovered.demand_rows and controller._fixed_phone_residency is None
    ):
        return discovered
    demand_rows = list(discovered.demand_rows)
    sessions = discovered.sessions
    helper_id = discovered.helper_id
    status = dict(discovered.route_evidence_by_artifact)
    for artifact_sha256, learning in _ranked_learning_demands(
        learning_by_artifact
    ):
        if any(row.manifest.artifact_sha256 == artifact_sha256 for row in demand_rows):
            continue
        identity = tuple(
            row.session_id for row in learning.sessions
        )
        if sessions is None:
            sessions = learning.sessions
            helper_id = learning.helper_id
        elif (
            helper_id != learning.helper_id
            or identity != tuple(row.session_id for row in sessions)
        ):
            status[artifact_sha256] = MappingProxyType({
                **dict(learning.status),
                "reason": "PHONE_RESIDENCY_SESSION_DOMAIN_MISMATCH",
            })
            continue
        demand_rows.append(learning.demand)
        status[artifact_sha256] = learning.status
    return _PhoneDemandDiscovery(
        demand_rows=tuple(demand_rows),
        sessions=sessions,
        helper_id=helper_id,
        route_evidence_by_artifact=status,
    )


def _ranked_learning_demands(
    learning_by_artifact: Mapping[str, _OfflineLearningDemand],
) -> tuple[tuple[str, _OfflineLearningDemand], ...]:
    return tuple(sorted(
        learning_by_artifact.items(),
        key=lambda row: (not row[1].release_priority, row[0]),
    ))


def _phone_route_used(controller, ticket: RuntimeRequestTicket) -> bool:
    plan = ticket.execution_plan
    if plan is not None and plan.execution_contract.phone_device_id is not None:
        return True
    for event in controller._model_placement_controller.request_helper_events(
        ticket.request.request_id
    ):
        fraction = event.get("selected_fraction_ppm")
        if (
            event.get("kind") in _PHONE_POLICY_EVENT_KINDS
            and type(fraction) is int
            and fraction > 0
        ):
            return True
    return False


def _record_phone_route_use(controller, ticket: RuntimeRequestTicket) -> None:
    """Remember, per artifact, whether the completed decision ran a phone route."""
    artifact_sha256 = ticket.model.artifact_sha256
    history = controller._phone_route_use_by_artifact.get(artifact_sha256, ())
    controller._phone_route_use_by_artifact[artifact_sha256] = (
        *history, _phone_route_used(controller, ticket)
    )[-controller._learning_demand_decision_window:]


def _learning_demand_active(controller, artifact_sha256: str) -> bool:
    """Cold start always learns; afterwards a phone route must have run within the window."""
    history = controller._phone_route_use_by_artifact.get(artifact_sha256, ())
    return (
        len(history) < controller._learning_demand_decision_window
        or any(history)
    )


def _learning_phone_demand(
    controller,
    candidate_set: AutomatedCandidateSet,
    request: Request,
    manifest: ModelManifest,
    queued_work: int,
) -> _OfflineLearningDemand | None:
    if not controller._learning_demand_active(manifest.artifact_sha256):
        return None
    allowed_rejections = {
        "COLD_RESIDENCY_BREAK_EVEN",
        "ENERGY_UNKNOWN",
        "MODEL_EPOCH_AUDIT_ONLY",
        "PHONE_RESIDENCY_LAYOUT_NOT_SELECTED",
        "ROUTE_MARGINAL_ENERGY_EVIDENCE_ABSENT",
        "ROUTE_NOT_QUALIFIED",
        *queued_demand_learning_rejections(controller),
    }
    candidate_by_id = {
        row.candidate_id: row for row in candidate_set.candidates
    }
    choices = []
    for opportunity in controller._compact_helper_opportunities(
        candidate_set, request
    ):
        if opportunity.evidence_state != "LEARNING":
            continue
        plan = opportunity.helper_operator_plan
        candidate = candidate_by_id.get(plan.route_id)
        phone_device_id = plan.execution_contract.phone_device_id
        helper = (
            None
            if phone_device_id is None else
            controller._runtime_capabilities.executor_by_device.get(
                phone_device_id
            )
        )
        shards = tuple(plan.execution_contract.phone_shards)
        physical_rejections = (
            set() if candidate is None else
            set(candidate.rejection_reasons) - allowed_rejections
        )
        phone_transitions = tuple(
            row for row in plan.transitions
            if phone_device_id in row.prepares_device_ids
        )
        if (
            candidate is None
            or physical_rejections
            or helper is None
            or not shards
            or not helper.phone_sessions
            or not phone_transitions
            or any(
                row.maturity != "QUALIFIED"
                for row in phone_transitions
            )
        ):
            continue
        sessions = tuple(sorted(
            helper.phone_sessions, key=lambda row: row.session_id
        ))
        session_by_id = {row.session_id: row for row in sessions}
        shard_session_ids = tuple(row.session_id for row in shards)
        if (
            len(shard_session_ids) != len(set(shard_session_ids))
            or not set(shard_session_ids).issubset(session_by_id)
            or any(
                row.endpoint != session_by_id[row.session_id].endpoint
                or row.resident_bytes
                    > session_by_id[row.session_id]
                        .resident_memory_limit_bytes
                for row in shards
            )
        ):
            continue
        layer_mask = 0
        for shard in shards:
            layer_mask |= shard.layer_mask
        operator_ids = []
        for operator in manifest.operators:
            if operator.kind != "ffn":
                continue
            prefix, separator, raw_index = operator.layer_id.partition(":")
            try:
                layer_index = int(raw_index)
            except ValueError:
                continue
            if (
                prefix == "layer"
                and separator == ":"
                and layer_mask & (1 << layer_index)
            ):
                operator_ids.append(operator.operator_id)
        columns = {row.maximum_columns for row in shards}
        if not operator_ids or len(columns) != 1:
            continue
        demand = PhoneFfnResidencyDemand(
            manifest=manifest,
            queued_work=queued_work,
            maximum_columns=next(iter(columns)),
            batch_plan=plan.execution_contract.batch_plan,
            benefit_by_operator={},
            benefit_value_kind="rough_compute_ops",
            allowed_operator_ids=tuple(sorted(operator_ids)),
        )
        status = MappingProxyType({
            "artifact_sha256": manifest.artifact_sha256,
            "candidate_rejections": {
                candidate.candidate_id: list(
                    candidate.rejection_reasons
                ),
            },
            "evidence_state": "LEARNING",
            "paired_desktop_route_id": (
                opportunity.desktop_parent_route_id
            ),
            "reason": "PHONE_RESIDENCY_LEARNING_EXPLORATION_READY",
            "source_route_id": candidate.candidate_id,
            "source_session_ids": [
                row.session_id for row in sessions
            ],
        })
        choices.append((
            len(shards),
            sum(row.resident_bytes for row in shards),
            candidate.candidate_id,
            _OfflineLearningDemand(
                demand=demand,
                sessions=sessions,
                helper_id=phone_device_id,
                status=status,
                release_priority=(
                    plan.adapter_parameters.get("ffn_host_share_release") == 1
                ),
            ),
        ))
    if not choices:
        return None
    return max(choices, key=lambda row: row[:3])[3]


def _online_learning_phone_discovery(
    controller,
    request: Request,
    manifest: ModelManifest,
    demand: _PhoneQueueDemand,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
    discovered: _PhoneDemandDiscovery,
) -> _PhoneDemandDiscovery:
    candidates_by_artifact = {
        manifest.artifact_sha256: (manifest.model_id, request),
    }
    for ticket in controller._runtime_controller.current_tickets():
        if ticket.dispatch_state in {"CANCELLED", "COMPLETED", "FAILED"}:
            continue
        candidates_by_artifact.setdefault(
            ticket.model.artifact_sha256,
            (ticket.model.model_id, ticket.request),
        )
    learning_by_artifact = {}
    for artifact_sha256, queued_work in sorted(
        demand.queued_work_by_artifact.items()
    ):
        status = discovered.route_evidence_by_artifact.get(
            artifact_sha256, {}
        )
        if status.get("reason") != "PHONE_RESIDENCY_ROUTE_EVIDENCE_UNUSABLE":
            continue
        owner = candidates_by_artifact.get(artifact_sha256)
        if owner is None:
            continue
        model_id, learning_request = owner
        learning_manifest = controller.runtime_model_manifest(model_id)
        candidate_set = controller._generate_automated_candidate_set(
            learning_request,
            learning_manifest,
            snapshot,
            observed_at_us,
            use_residency_holds=False,
            update_phone_residency_portfolio=False,
        )
        learning = controller._learning_phone_demand(
            candidate_set,
            learning_request,
            learning_manifest,
            queued_work,
        )
        if learning is not None:
            learning_by_artifact[artifact_sha256] = learning
            controller._online_learning_phone_demand_cache[
                artifact_sha256
            ] = learning
    return controller._merge_learning_phone_discovery(
        discovered, learning_by_artifact
    )


def _cached_online_learning_phone_discovery(
    controller,
    demand: _PhoneQueueDemand,
    discovered: _PhoneDemandDiscovery,
) -> _PhoneDemandDiscovery:
    learning_by_artifact = {}
    for artifact_sha256, queued_work in sorted(
        demand.queued_work_by_artifact.items()
    ):
        status = discovered.route_evidence_by_artifact.get(
            artifact_sha256, {}
        )
        cached = controller._online_learning_phone_demand_cache.get(
            artifact_sha256
        )
        if (
            status.get("reason")
                != "PHONE_RESIDENCY_ROUTE_EVIDENCE_UNUSABLE"
            or not isinstance(cached, _OfflineLearningDemand)
        ):
            continue
        if not controller._learning_demand_active(artifact_sha256):
            controller._online_learning_phone_demand_cache.pop(
                artifact_sha256, None
            )
            continue
        helper = controller._runtime_capabilities.executor_by_device.get(
            cached.helper_id
        )
        current_sessions = (
            () if helper is None else tuple(sorted(
                helper.phone_sessions, key=lambda row: row.session_id
            ))
        )
        if current_sessions != cached.sessions:
            controller._online_learning_phone_demand_cache.pop(
                artifact_sha256, None
            )
            continue
        learning_by_artifact[artifact_sha256] = replace(
            cached,
            demand=replace(cached.demand, queued_work=queued_work),
        )
    return controller._merge_learning_phone_discovery(
        discovered, learning_by_artifact
    )


def _merge_learning_phone_discovery(
    discovered: _PhoneDemandDiscovery,
    learning_by_artifact: Mapping[str, _OfflineLearningDemand],
) -> _PhoneDemandDiscovery:
    if not learning_by_artifact:
        return discovered
    rows = []
    sessions = None
    helper_id = None
    statuses = dict(discovered.route_evidence_by_artifact)
    for artifact_sha256, learning in _ranked_learning_demands(
        learning_by_artifact
    ):
        session_ids = tuple(
            row.session_id for row in learning.sessions
        )
        if sessions is None:
            sessions = learning.sessions
            helper_id = learning.helper_id
        elif (
            helper_id != learning.helper_id
            or session_ids != tuple(
                row.session_id for row in sessions
            )
        ):
            statuses[artifact_sha256] = MappingProxyType({
                **dict(learning.status),
                "reason": "PHONE_RESIDENCY_SESSION_DOMAIN_MISMATCH",
            })
            continue
        rows.append(learning.demand)
        statuses[artifact_sha256] = learning.status
    return _PhoneDemandDiscovery(
        demand_rows=tuple(rows),
        sessions=sessions,
        helper_id=helper_id,
        route_evidence_by_artifact=statuses,
    )


def _phone_queue_demand(
    controller,
    request: Request,
    manifest: ModelManifest,
) -> _PhoneQueueDemand:
    demand_by_request_id = {}
    for ticket in controller._runtime_controller.current_tickets():
        if ticket.dispatch_state in {
            "CANCELLED", "COMPLETED", "FAILED"
        }:
            continue
        active = ticket.dispatch_state == "ACQUIRED"
        remaining_tokens = (
            controller._model_placement_controller
            .remaining_request_decode_tokens(
                ticket.request.request_id,
                ticket.request.output_tokens,
            )
            if active else ticket.request.output_tokens
        )
        demand_by_request_id[ticket.request.request_id] = (
            ticket.model.artifact_sha256,
            "ACTIVE" if active else "QUEUED",
            remaining_tokens,
        )
    demand_by_request_id.setdefault(
        request.request_id,
        (manifest.artifact_sha256, "QUEUED", request.output_tokens),
    )
    active_count: dict[str, int] = {}
    queued_count: dict[str, int] = {}
    active_tokens: dict[str, int] = {}
    queued_tokens: dict[str, int] = {}
    for artifact_sha256, state, remaining_tokens in (
        demand_by_request_id.values()
    ):
        counts = active_count if state == "ACTIVE" else queued_count
        counts[artifact_sha256] = counts.get(artifact_sha256, 0) + 1
        tokens = active_tokens if state == "ACTIVE" else queued_tokens
        tokens[artifact_sha256] = (
            tokens.get(artifact_sha256, 0) + remaining_tokens
        )
    return _PhoneQueueDemand(
        active_count_by_artifact=active_count,
        queued_count_by_artifact=queued_count,
        active_remaining_tokens_by_artifact=active_tokens,
        queued_output_tokens_by_artifact=queued_tokens,
        queued_work_by_artifact=controller._arrived_decode_work_by_artifact(
            active_tokens, queued_tokens
        ),
    )


def _discover_phone_residency_demand(
    controller,
    compiler,
    queued_work_by_artifact: Mapping[str, int],
) -> _PhoneDemandDiscovery:
    manifest_by_artifact = {
        row.artifact_sha256: row
        for row in controller._runtime_manifests.values()
    }
    demand_rows = []
    route_evidence = {}
    discovered_sessions = None
    helper_id = None
    for artifact_sha256, queued_work in sorted(
        queued_work_by_artifact.items()
    ):
        queued_manifest = manifest_by_artifact.get(artifact_sha256)
        if queued_manifest is None:
            continue
        prepared = compiler.phone_residency_demand(
            queued_manifest, queued_work
        )
        route_evidence[artifact_sha256] = (
            compiler.phone_residency_evidence_status(artifact_sha256)
        )
        if prepared is None:
            continue
        demand, sessions, demand_helper_id = prepared
        session_identity = tuple(row.session_id for row in sessions)
        if discovered_sessions is None:
            discovered_sessions = tuple(sessions)
            helper_id = demand_helper_id
        elif (
            helper_id != demand_helper_id
            or session_identity != tuple(
                row.session_id for row in discovered_sessions
            )
        ):
            route_evidence[artifact_sha256] = MappingProxyType({
                **dict(route_evidence[artifact_sha256]),
                "reason": "PHONE_RESIDENCY_SESSION_DOMAIN_MISMATCH",
            })
            continue
        demand_rows.append(demand)
    return _PhoneDemandDiscovery(
        demand_rows=tuple(demand_rows),
        sessions=discovered_sessions,
        helper_id=helper_id,
        route_evidence_by_artifact=route_evidence,
    )
