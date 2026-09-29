"""Derive safe decode probe policies from generated runtime candidates."""

from __future__ import annotations

from dataclasses import replace

from .adaptive_decode_contracts import (
    AdaptiveDecodeError,
    AdaptiveDecodePolicy,
)
from .model_manifest import ModelManifest
from .runtime_capabilities import RuntimeCapabilityCatalog
from .runtime_plan import AutomatedCandidateSet, AutomatedRouteCandidate


def _layer_index(layer_id: str) -> int:
    prefix, separator, suffix = layer_id.partition(":")
    if prefix != "layer" or separator != ":":
        raise AdaptiveDecodeError("adaptive FFN layer identity is invalid")
    try:
        result = int(suffix)
    except ValueError as error:
        raise AdaptiveDecodeError(
            "adaptive FFN layer identity is invalid"
        ) from error
    if not 0 <= result < 64:
        raise AdaptiveDecodeError("adaptive FFN layer exceeds runtime mask")
    return result


def _per_token(value: int | None, output_tokens: int) -> int | None:
    if value is None:
        return None
    return max(1, value // max(1, output_tokens))


def _same_operator_placement(left: object, right: object) -> bool:
    return all(
        getattr(left, name) == getattr(right, name)
        for name in (
            "operator_id",
            "operator_kind",
            "device_ids",
            "split_axis",
            "split_fraction_ppm",
            "kernel_profile_ids",
        )
    )


def _policy(
    candidate: AutomatedRouteCandidate,
    parent: AutomatedRouteCandidate,
    manifest: ModelManifest,
    output_tokens: int,
) -> AdaptiveDecodePolicy:
    parent_by_operator = {
        row.operator_id: row for row in parent.plan.operators
    }
    manifest_by_operator = {
        row.operator_id: row for row in manifest.operators
    }
    if (
        set(parent_by_operator) != set(manifest_by_operator)
        or {row.operator_id for row in candidate.plan.operators}
            != set(manifest_by_operator)
    ):
        raise AdaptiveDecodeError(
            "adaptive plan operator identity differs from the model"
        )
    changed_layers = []
    for assignment in candidate.plan.operators:
        operator = manifest_by_operator[assignment.operator_id]
        parent_assignment = parent_by_operator[assignment.operator_id]
        if (
            assignment.operator_kind != operator.kind
            or parent_assignment.operator_kind != operator.kind
        ):
            raise AdaptiveDecodeError(
                "adaptive plan operator kind differs from the model"
            )
        changed = not _same_operator_placement(
            assignment, parent_assignment
        )
        if changed and operator.kind != "ffn":
            raise AdaptiveDecodeError(
                "adaptive decode may change only stateless FFN placement"
            )
        if changed:
            changed_layers.append(_layer_index(operator.layer_id))
    layers = tuple(sorted(set(changed_layers)))
    if not layers:
        raise AdaptiveDecodeError("adaptive candidate has no changed FFN")
    if candidate.route_family == "operator_offload":
        columns = manifest.feed_forward_length
        fraction = 1_000_000
    elif (
        candidate.route_family == "operator_split"
        and candidate.split_axis == "column"
    ):
        scaled = (
            manifest.feed_forward_length * candidate.split_fraction_ppm
        )
        if scaled % 1_000_000:
            raise AdaptiveDecodeError(
                "adaptive FFN split does not end at a physical column"
            )
        columns = scaled // 1_000_000
        fraction = candidate.split_fraction_ppm
    else:
        raise AdaptiveDecodeError("adaptive candidate is not an FFN column route")
    return AdaptiveDecodePolicy(
        route_id=candidate.candidate_id,
        executor_id=candidate.binding.executor_id,
        operator_plan_sha256=candidate.plan.plan_sha256,
        desktop_parent_route_id=parent.candidate_id,
        desktop_placement_sha256=parent.plan.desktop_placement_sha256,
        layer_indices=layers,
        layer_mask=sum(1 << value for value in layers),
        columns=columns,
        split_fraction_ppm=fraction,
        resource_ids=candidate.plan.resource_ids,
        baseline=False,
        predicted_latency_per_token_us=_per_token(
            candidate.cost.service_us, output_tokens
        ),
        predicted_energy_per_token_uj=_per_token(
            candidate.cost.fleet_energy_uj, output_tokens
        ),
    )


def _envelope_subpolicies(
    envelope_policy: AdaptiveDecodePolicy,
    envelope: AutomatedRouteCandidate,
    manifest: ModelManifest,
    catalog: RuntimeCapabilityCatalog,
) -> tuple[AdaptiveDecodePolicy, ...]:
    parameters = envelope.plan.adapter_parameters
    coordinator = catalog.composite_executor_by_id.get(
        envelope.binding.executor_id
    )
    quantum = parameters.get("ffn_column_quantum")
    if (
        coordinator is None
        or coordinator.split_axis != "column"
        or parameters.get("ffn_assistance_phase") != "decode"
        or parameters.get("ffn_runtime_control_protocol")
            != "decode-boundary-v1"
        or parameters.get("ffn_weight_buffer_layout")
            != "resident-superset"
        or type(quantum) is not int
        or quantum <= 0
    ):
        return ()
    resident_columns = parameters.get("ffn_resident_columns")
    resident_layer_mask = parameters.get("ffn_resident_layer_mask")
    if (
        type(resident_columns) is not int
        or resident_columns <= 0
        or resident_columns > manifest.feed_forward_length
        or type(resident_layer_mask) is not int
        or resident_layer_mask <= 0
        or resident_layer_mask >> manifest.block_count
    ):
        return ()
    resident_layers = tuple(
        index for index in range(manifest.block_count)
        if resident_layer_mask & (1 << index)
    )
    rows = []
    target_fractions = set(coordinator.split_fractions_ppm)
    target_fractions.update(
        62_500 * index for index in range(1, 17)
    )
    target_fractions.add(envelope_policy.split_fraction_ppm)
    by_columns = {}
    for fraction in sorted(target_fractions):
        target_columns = (
            manifest.feed_forward_length * fraction + 500_000
        ) // 1_000_000
        columns = (
            (target_columns + quantum // 2) // quantum
        ) * quantum
        if (
            columns <= 0
            or columns > resident_columns
        ):
            continue
        actual_fraction = (
            columns * 1_000_000 // manifest.feed_forward_length
        )
        if (
            columns == envelope_policy.columns
            and actual_fraction == envelope_policy.split_fraction_ppm
            and resident_layer_mask == envelope_policy.layer_mask
        ):
            continue
        by_columns[columns] = AdaptiveDecodePolicy(
            route_id=(
                envelope.candidate_id
                + ":decode-window:"
                + str(actual_fraction)
            ),
            executor_id=envelope_policy.executor_id,
            operator_plan_sha256=envelope_policy.operator_plan_sha256,
            desktop_parent_route_id=(
                envelope_policy.desktop_parent_route_id
            ),
            desktop_placement_sha256=(
                envelope_policy.desktop_placement_sha256
            ),
            layer_indices=resident_layers,
            layer_mask=resident_layer_mask,
            columns=columns,
            split_fraction_ppm=actual_fraction,
            resource_ids=envelope_policy.resource_ids,
        )
    rows.extend(by_columns.values())
    return tuple(sorted(
        rows,
        key=lambda row: (row.split_fraction_ppm, row.policy_hash),
    ))


def adaptive_desktop_control_is_qualified(
    candidate_set: AutomatedCandidateSet,
    catalog: RuntimeCapabilityCatalog,
) -> bool:
    baseline = candidate_set.baseline
    control = catalog.desktop_control_by_artifact.get(
        baseline.binding.artifact_sha256
    )
    coordinator = catalog.composite_executor_by_id.get(
        baseline.binding.executor_id
    )
    return bool(
        control is not None
        and coordinator is not None
        and control.maturity == "QUALIFIED"
        and control.executor_id == baseline.binding.executor_id
        and control.operator_placements == coordinator.operator_placements
        and control.placement_sha256 == baseline.plan.desktop_placement_sha256
    )


def adaptive_candidate_set_for_parent(
    candidate_set: AutomatedCandidateSet,
    parent: AutomatedRouteCandidate,
) -> AutomatedCandidateSet:
    """Scope helper derivation without changing the route's qualification."""

    if parent.candidate_id == candidate_set.baseline_route_id:
        return candidate_set
    if (
        parent not in candidate_set.candidates
        or parent.plan.execution_contract.execution_mode != "desktop"
        or parent.plan.baseline_executor_id is not None
    ):
        raise AdaptiveDecodeError("adaptive execution parent is not a desktop row")
    return replace(
        candidate_set,
        baseline_route_id=parent.candidate_id,
        candidates=tuple(
            replace(row, baseline=row.candidate_id == parent.candidate_id)
            for row in candidate_set.candidates
        ),
    )


def adaptive_decode_policies(
    candidate_set: AutomatedCandidateSet,
    manifest: ModelManifest,
    catalog: RuntimeCapabilityCatalog,
    output_tokens: int,
    maximum_phone_sessions: int | None = None,
) -> tuple[
    AdaptiveDecodePolicy,
    tuple[AdaptiveDecodePolicy, ...],
    AutomatedRouteCandidate | None,
]:
    """Return paired FFN policies and the physical maximum-slice envelope."""
    if (
        not isinstance(candidate_set, AutomatedCandidateSet)
        or not isinstance(manifest, ModelManifest)
        or not isinstance(catalog, RuntimeCapabilityCatalog)
        or type(output_tokens) is not int
        or output_tokens <= 0
        or (
            maximum_phone_sessions is not None
            and (
                type(maximum_phone_sessions) is not int
                or maximum_phone_sessions <= 0
            )
        )
    ):
        raise AdaptiveDecodeError("adaptive planning input is invalid")
    baseline = candidate_set.baseline
    placement = baseline.plan.desktop_placement_sha256
    if placement is None:
        raise AdaptiveDecodeError("adaptive desktop placement is unbound")
    device_kinds = {
        device_id: catalog.placement_profile.devices[device_id].kind
        for device_id in baseline.plan.device_ids
    }
    if not device_kinds or set(device_kinds.values()) - {"cpu", "gpu"}:
        raise AdaptiveDecodeError(
            "adaptive baseline must execute on desktop devices"
        )
    baseline_policy = AdaptiveDecodePolicy(
        route_id=baseline.candidate_id,
        executor_id=baseline.binding.executor_id,
        operator_plan_sha256=baseline.plan.plan_sha256,
        desktop_parent_route_id=baseline.candidate_id,
        desktop_placement_sha256=placement,
        layer_indices=(),
        layer_mask=0,
        columns=0,
        split_fraction_ppm=0,
        resource_ids=baseline.plan.resource_ids,
        baseline=True,
        predicted_latency_per_token_us=_per_token(
            baseline.cost.service_us, output_tokens
        ),
        predicted_energy_per_token_uj=_per_token(
            baseline.cost.fleet_energy_uj, output_tokens
        ),
    )
    by_id = {row.candidate_id: row for row in candidate_set.candidates}
    baseline_coordinator = catalog.composite_executor_by_id.get(
        baseline.binding.executor_id
    )
    desktop_control_qualified = adaptive_desktop_control_is_qualified(
        candidate_set, catalog
    )
    learning_parent = (
        baseline_coordinator is not None
        and baseline_coordinator.maturity == "QUALIFIED"
        and baseline_coordinator.artifact_sha256 == manifest.artifact_sha256
        and baseline.plan.baseline_executor_id is None
        and baseline.plan.execution_contract.execution_mode == "desktop"
    )
    allowed_rejections = {
        "COLD_RESIDENCY_BREAK_EVEN",
        "ENERGY_UNKNOWN",
        "MARGINAL_SYSTEM_COST_UNKNOWN",
        "MODEL_EPOCH_AUDIT_ONLY",
        "PHONE_RESIDENCY_LAYOUT_NOT_SELECTED",
        "ROUTE_MARGINAL_ENERGY_EVIDENCE_ABSENT",
        "ROUTE_NOT_QUALIFIED",
        "SLO_UPPER_BOUND",
    }
    rows: list[tuple[AdaptiveDecodePolicy, AutomatedRouteCandidate]] = []
    for candidate in candidate_set.candidates:
        if (
            (
                maximum_phone_sessions is not None
                and len(candidate.plan.execution_contract.phone_shards)
                    > maximum_phone_sessions
            )
            or
            candidate.assisted_operator_kind != "ffn"
            or candidate.route_family != "operator_split"
            or candidate.paired_baseline_route_id is None
            or candidate.paired_baseline_route_id not in by_id
            or set(candidate.rejection_reasons) - allowed_rejections
        ):
            continue
        parent = by_id[candidate.paired_baseline_route_id]
        coordinator = catalog.composite_executor_by_id.get(
            candidate.binding.executor_id
        )
        capabilities = tuple(
            catalog.executor_by_device[device_id]
            for device_id in candidate.device_ids
        )
        if (
            parent.candidate_id != baseline.candidate_id
            or not (desktop_control_qualified or learning_parent)
            or candidate.plan.desktop_placement_sha256 != placement
            or coordinator is None
            or (
                coordinator.maturity != "QUALIFIED"
                and candidate.maturity != "QUALIFIED"
            )
            or candidate.binding.endpoint is None
            or candidate.binding.operator_plan_protocol is None
            or any(row.maturity != "QUALIFIED" for row in capabilities)
            or any(
                transition.maturity != "QUALIFIED"
                for transition in candidate.plan.transitions
            )
        ):
            continue
        try:
            rows.append((
                _policy(candidate, parent, manifest, output_tokens),
                candidate,
            ))
        except AdaptiveDecodeError:
            continue
    if not rows:
        return baseline_policy, (), None
    rows = [
        (policy, candidate)
        for policy, candidate in rows
        if policy.layer_mask == candidate.plan.adapter_parameters.get(
            "ffn_resident_layer_mask"
        )
    ]
    if not rows:
        return baseline_policy, (), None
    maximum_geometry = max(
        (len(policy.layer_indices), policy.columns)
        for policy, _candidate in rows
    )
    envelope_policy, envelope = min(
        (
            (policy, candidate)
            for policy, candidate in rows
            if (len(policy.layer_indices), policy.columns)
                == maximum_geometry
        ),
        key=lambda row: (
            row[1].cost.fleet_energy_upper_uj
            if row[1].cost.energy_evidence in {
                "CALIBRATED", "MEASURED"
            }
            else 2**63 - 1,
            row[1].cost.service_upper_us,
            row[1].cost.service_us,
            row[0].policy_hash,
        ),
    )
    resident_mask = int(
        envelope.plan.adapter_parameters["ffn_resident_layer_mask"]
    )
    resident_columns = int(
        envelope.plan.adapter_parameters["ffn_resident_columns"]
    )
    policies = tuple(sorted(
        (
            policy for policy, candidate in rows
            if candidate.binding.executor_id
                == envelope.binding.executor_id
            if policy.layer_mask == resident_mask
            and policy.columns <= resident_columns
        ),
        key=lambda row: (row.columns, len(row.layer_indices), row.route_id),
    ))
    by_geometry = {
        (policy.layer_mask, policy.columns): policy
        for policy in policies
    }
    for policy in _envelope_subpolicies(
        envelope_policy, envelope, manifest, catalog
    ):
        by_geometry.setdefault(
            (policy.layer_mask, policy.columns), policy
        )
    by_geometry[
        (envelope_policy.layer_mask, envelope_policy.columns)
    ] = envelope_policy
    policies = tuple(sorted(
        by_geometry.values(),
        key=lambda row: (row.columns, len(row.layer_indices), row.route_id),
    ))
    return baseline_policy, policies, envelope


def adaptive_probe_contracts(
    candidate_set: AutomatedCandidateSet,
    manifest: ModelManifest,
    catalog: RuntimeCapabilityCatalog,
    output_tokens: int,
) -> dict[str, dict[str, object]]:
    result = _adaptive_parent_probe_contracts(
        candidate_set, manifest, catalog, output_tokens
    )
    for parent in candidate_set.candidates:
        if (
            parent.candidate_id != candidate_set.baseline_route_id
            and parent.admitted
            and parent.plan.execution_contract.execution_mode == "desktop"
            and all(
                catalog.placement_profile.devices[device_id].kind
                    in {"cpu", "gpu"}
                for device_id in parent.plan.device_ids
            )
            and (parent.plan.helper_envelope is not None or
                 parent.plan.adapter_parameters.get("dormant_phone_ffn_runtime_v1"))
        ):
            result.update(_adaptive_parent_probe_contracts(
                adaptive_candidate_set_for_parent(candidate_set, parent),
                manifest, catalog, output_tokens,
            ))
    return result


def _adaptive_parent_probe_contracts(
    candidate_set: AutomatedCandidateSet,
    manifest: ModelManifest,
    catalog: RuntimeCapabilityCatalog,
    output_tokens: int,
) -> dict[str, dict[str, object]]:
    baseline, policies, envelope = adaptive_decode_policies(
        candidate_set, manifest, catalog, output_tokens
    )
    result = {baseline.route_id: baseline.to_json()}
    result.update({row.route_id: row.to_json() for row in policies})
    if envelope is not None:
        result[envelope.candidate_id] = {
            **result[envelope.candidate_id],
            "adaptive_envelope": True,
            "adaptive_decode_probe_contracts": [
                row.to_json() for row in policies
            ],
        }
    by_id = {row.candidate_id: row for row in candidate_set.candidates}
    for candidate in candidate_set.candidates:
        execution_contract = candidate.plan.execution_contract
        if execution_contract.execution_mode != "adaptive-split":
            continue
        parent_id = candidate.paired_baseline_route_id
        if parent_id is None or parent_id not in by_id:
            continue
        parent = by_id[parent_id]
        if parent.candidate_id != baseline.route_id:
            continue
        try:
            selected = _policy(
                candidate, parent, manifest, output_tokens
            )
        except AdaptiveDecodeError:
            continue
        by_geometry = {
            (selected.layer_mask, selected.columns): selected
        }
        for policy in _envelope_subpolicies(
            selected, candidate, manifest, catalog
        ):
            by_geometry.setdefault(
                (policy.layer_mask, policy.columns), policy
            )
        allowed = set(
            execution_contract.allowed_adaptive_fractions_ppm
        )
        candidate_policies = tuple(sorted(
            (
                policy for policy in by_geometry.values()
                if policy.split_fraction_ppm in allowed
            ),
            key=lambda row: (
                row.split_fraction_ppm,
                row.layer_mask,
                row.policy_hash,
            ),
        ))
        represented = {
            policy.split_fraction_ppm for policy in candidate_policies
        }
        if represented != allowed - {0}:
            continue
        result[candidate.candidate_id] = {
            **selected.to_json(),
            "adaptive_envelope": True,
            "adaptive_decode_probe_contracts": [
                row.to_json() for row in candidate_policies
            ],
        }
    return result
