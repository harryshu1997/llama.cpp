#!/usr/bin/env python3
"""Run the 74-request F16 trace plus ten-request Llama overlay."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import traceback
from typing import Any, Mapping, Sequence
from urllib.parse import SplitResult, urlsplit


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    AdaptiveDecodeConfig,
    ModelManifest,
    PhoneFfnShardStorageMetadata,
    Request,
    RuntimeCapabilityCatalog,
    RuntimeDispatchPolicy,
    UnifiedScheduler,
)
from research_dev.scheduler.adapters.ffn_shards import (  # noqa: E402
    FfnShardIndex,
    FfnShardIndexError,
)
from research_dev.scheduler.adapters import (  # noqa: E402
    AndroidLlamaServerProcessConfiguration,
    CanonicalArrivalCoordinator,
    CanonicalPhysicalAdapter,
    CanonicalOfflinePhoneResidencyPreloader,
    CanonicalRuntimeSubmission,
    default_host_metric_callbacks,
    DirectPhoneFfnSessionConfiguration,
    HeterogeneousPhysicalRig,
    HeterogeneousRigConfiguration,
    LlamaCppCompletionPayload,
    load_transport_qualification_identity,
    validate_decision_candidate_coverage,
)
from research_dev.scheduler.config import (  # noqa: E402
    FixedPhoneResidencyConfiguration, PhoneHtpMemoryCapConfiguration,
    PhoneResidentModelReprovisioningConfiguration, PhoneThermalStatusLimitConfiguration,
    StartupDesktopParentConfiguration,
)
from research_dev.scheduler.campaigns.burstgpt.common import (  # noqa: E402
    CONFIRMATION as CONFIRMATION,
    UnifiedTraceError as UnifiedTraceError,
    require as require,
    canonical as canonical,
    digest as digest,
    load_object as load_object,
    load_rows as load_rows,
)
from research_dev.scheduler.campaigns.burstgpt.arguments import (  # noqa: E402
    host_dependency as host_dependency,
    _build_parser as _build_parser,
    _validate_arguments as _validate_arguments,
)
from research_dev.scheduler.campaigns.burstgpt.trace_inputs import (  # noqa: E402
    REPLAY_SCHEDULE_SCHEMA as REPLAY_SCHEDULE_SCHEMA,
    REPLAY_START_US as REPLAY_START_US,
    QWEN_ROLE as QWEN_ROLE,
    GEMMA_ROLE as GEMMA_ROLE,
    TRACE_HOT_MODEL_ID as TRACE_HOT_MODEL_ID,
    TRACE_COLD_MODEL_ID as TRACE_COLD_MODEL_ID,
    trace_role as trace_role,
    validate_trace as validate_trace,
    merge_rows as merge_rows,
    select_rows as select_rows,
    scale_replay_arrivals as scale_replay_arrivals,
    apply_named_replay_schedule as apply_named_replay_schedule,
    persist_replay_schedule as persist_replay_schedule,
)

# canonical re-exports kept for campaign tooling (test_campaign_inputs enforces them)
__all__ = [
    'CONFIRMATION',
    'UnifiedTraceError',
    'require',
    'canonical',
    'digest',
    'load_object',
    'load_rows',
    'host_dependency',
    '_build_parser',
    '_validate_arguments',
    'REPLAY_SCHEDULE_SCHEMA',
    'REPLAY_START_US',
    'QWEN_ROLE',
    'GEMMA_ROLE',
    'TRACE_HOT_MODEL_ID',
    'TRACE_COLD_MODEL_ID',
    'trace_role',
    'validate_trace',
    'merge_rows',
    'select_rows',
    'scale_replay_arrivals',
    'apply_named_replay_schedule',
    'persist_replay_schedule',
]


RESULT_SCHEMA = "s42-unified-fp16-llama-overlay-result-v1"

UnifiedFp16PhysicalRig = HeterogeneousPhysicalRig
UnifiedFp16RigConfiguration = HeterogeneousRigConfiguration


def execution_fraction_history(
    execution_contract: object | None,
    ticket_id: str,
    adaptive_groups: list[dict[str, Any]],
    phone_call_count: int,
    *,
    request_id: str | None = None,
    decode_cohort: object | None = None,
) -> dict[str, object]:
    initial_fraction = (
        0
        if execution_contract is None
        else int(execution_contract.initial_split_fraction_ppm)
    )
    execution_mode = (
        "desktop"
        if execution_contract is None
        else str(execution_contract.execution_mode)
    )
    matches = [
        group for group in adaptive_groups
        if group.get("ticket_id") == ticket_id
    ]
    require(len(matches) <= 1, "adaptive execution group is ambiguous")
    observation_scope = "request"
    windows_override: list[dict[str, Any]] | None = None
    if execution_mode == "adaptive-split" and not matches:
        require(
            request_id is not None and decode_cohort is not None,
            "adaptive execution lacks its ticket-bound observation",
        )
        cohort_id = str(decode_cohort.cohort_id)
        leader_request_id = str(decode_cohort.leader_request_id)
        require(
            request_id in decode_cohort.member_request_ids,
            "adaptive execution differs from its decode cohort",
        )
        cohort_matches: list[
            tuple[dict[str, Any], list[dict[str, Any]]]
        ] = []
        for group in adaptive_groups:
            if group.get("request_id") != leader_request_id:
                continue
            group_windows = group.get("windows")
            if type(group_windows) is not list:
                continue
            shared_windows = [
                window for window in group_windows
                if window.get("cohort_id") == cohort_id
                and request_id
                    in (window.get("cohort_member_request_ids") or ())
                and window.get("energy_owner_request_id")
                    == leader_request_id
            ]
            if shared_windows:
                cohort_matches.append((group, shared_windows))
        require(
            len(cohort_matches) == 1,
            "adaptive cohort execution observation is missing or ambiguous",
        )
        matches = [cohort_matches[0][0]]
        windows_override = cohort_matches[0][1]
        observation_scope = "decode_cohort_shared"
    require(
        execution_mode != "adaptive-split" or len(matches) == 1,
        "adaptive execution lacks its ticket-bound observation",
    )
    if not matches:
        return {
            "explored_split_fractions_ppm": [],
            "initial_split_fraction_ppm": initial_fraction,
            "phone_executed_split_fractions_ppm": (
                [initial_fraction]
                if phone_call_count > 0 and initial_fraction > 0 else []
            ),
            "physically_executed_split_fractions_ppm": [initial_fraction],
            "selected_split_fraction_ppm": initial_fraction,
        }
    group = matches[0]
    windows = (
        group.get("windows")
        if windows_override is None else windows_override
    )
    final_policy = group.get("final_policy")
    require(
        type(windows) is list and type(final_policy) is dict,
        "adaptive execution group is invalid",
    )
    explored_fractions = sorted({
        int(window["policy"]["split_fraction_ppm"])
        for window in windows
        if window.get("window_role") == "exploration"
    })
    executed_fractions = sorted({
        int(window["policy"]["split_fraction_ppm"])
        for window in windows
        if window.get("failure_reason") is None
        and window.get("output_valid") is True
    })
    phone_executed_fractions = sorted({
        int(window["policy"]["split_fraction_ppm"])
        for window in windows
        if int(window.get("completed_phone_calls") or 0) > 0
    })
    selected_fraction = int(final_policy["split_fraction_ppm"])
    if observation_scope == "decode_cohort_shared":
        require(windows, "adaptive cohort execution has no shared window")
        selected_fraction = int(windows[-1]["policy"]["split_fraction_ppm"])
    return {
        "adaptive_grouped_observation_sha256": group.get(
            "grouped_observation_sha256"
        ),
        "adaptive_observation_scope": observation_scope,
        "adaptive_observation_ticket_id": group.get("ticket_id"),
        "explored_split_fractions_ppm": explored_fractions,
        "initial_split_fraction_ppm": initial_fraction,
        "phone_executed_split_fractions_ppm": phone_executed_fractions,
        "physically_executed_split_fractions_ppm": executed_fractions,
        "selected_split_fraction_ppm": selected_fraction,
    }


def artifact_aliases(
    catalog: RuntimeCapabilityCatalog,
    manifests: dict[str, ModelManifest],
) -> dict[str, str]:
    result = {}
    for model_id, manifest in manifests.items():
        artifact_candidates = {
            row.adapter_parameters.get("model_alias")
            for row in catalog.composite_executors
            if row.artifact_sha256 == manifest.artifact_sha256
            and type(row.adapter_parameters.get("model_alias")) is str
        }
        exact_base_candidates = {
            row.adapter_parameters.get("model_alias")
            for row in catalog.executors
            if row.adapter_parameters
            and row.adapter_parameters.get("model_alias") == model_id
        }
        aliases = artifact_candidates or exact_base_candidates
        require(len(aliases) == 1, f"model alias: {model_id}")
        result[model_id] = aliases.pop()
    return result


def measured_energy(value) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "attribution_kind": value.attribution_kind,
        "energy_boundary_id": value.energy_boundary_id,
        "fleet_energy_uj_by_domain": dict(
            value.fleet_energy_uj_by_domain
        ),
        "measurement_evidence_ids": list(value.measurement_evidence_ids),
        "transfer_energy_uj_by_link": dict(
            value.transfer_energy_uj_by_link
        ),
        "estimation_metadata": dict(value.estimation_metadata),
    }


def trace_energy_accounting(value) -> dict[str, object]:
    measured = measured_energy(value)
    require(measured is not None, "trace-level fleet energy is absent")
    return {
        "authoritative_total_field": "trace_energy",
        "boundary": "one paid trace interval",
        "comparison_scope": "trace_level_whole_fleet",
        "per_request_energy_role": "diagnostic_only",
        "per_request_windows_additive": False,
    }


def candidate_device_families(records: list[dict[str, Any]]) -> set[tuple[str, ...]]:
    return {
        tuple(sorted(
            participant["device_id"]
            for participant in candidate["executor"]["participants"]
        ))
        for record in records
        if record["event_kind"] == "DECISION"
        for candidate in record["candidates"]
        if candidate.get("executor") is not None
    }


def validate_attempt_chains(
    attempts: list[dict[str, Any]], request_ids: set[str]
) -> None:
    by_request = {request_id: [] for request_id in request_ids}
    for row in attempts:
        request_id = row["request_ids"][0]
        require(request_id in by_request, "unknown scheduler attempt")
        by_request[request_id].append(row)
    for request_id, rows in by_request.items():
        rows.sort(key=lambda row: row["attempt_index"])
        require(
            rows
            and rows[0]["event_kind"] == "DECISION"
            and rows[0]["attempt_index"] == 0
            and rows[0]["previous_ticket_id"] is None,
            "initial scheduler attempt: " + request_id,
        )
        for index, row in enumerate(rows[1:], 1):
            require(
                row["event_kind"] in {"REPLAN", "FALLBACK"}
                and row["attempt_index"] == index
                and row["previous_ticket_id"] == rows[index - 1]["ticket_id"],
                "scheduler attempt chain: " + request_id,
            )


@dataclass(frozen=True)
class _TraceModels:
    trace_manifest: dict[str, Any]
    large_rows: list[dict[str, Any]]
    overlay_rows: list[dict[str, Any]]
    expected_qwen: ModelManifest
    expected_gemma: ModelManifest
    llama_model_id: str
    model_paths: dict[str, Path]
    catalog: RuntimeCapabilityCatalog
    diagnostic_url: SplitResult


def _apply_phone_thermal_status_limits(
    catalog: RuntimeCapabilityCatalog, text: str | None
) -> RuntimeCapabilityCatalog:
    """Campaign `phone_thermal_status_limits` (opt-in per-phone Android thermal status limit)
    override the materialized catalog executors; absent keeps the catalog unchanged."""
    if text is None:
        return catalog
    rows = json.loads(text)
    require(isinstance(rows, list), "phone thermal status limits must be an array")
    configured = tuple(PhoneThermalStatusLimitConfiguration.from_json(row) for row in rows)
    limits = {row.phone_device_id: row.maximum_thermal_status for row in configured}
    require(len(limits) == len(configured), "phone thermal status limit devices are not unique")
    devices = catalog.placement_profile.devices
    require(
        all(
            device_id in catalog.executor_by_device
            and devices[device_id].kind == "phone"
            for device_id in limits
        ),
        "phone thermal status limit device is not a catalog phone",
    )
    return replace(catalog, executors=tuple(
        replace(row, maximum_thermal_status=limits[row.device_id])
        if row.device_id in limits else row
        for row in catalog.executors
    ))


def _load_trace_models(args: argparse.Namespace) -> _TraceModels:
    trace_manifest = load_object(args.trace_manifest)
    large_rows, overlay_rows = validate_trace(
        args.large_requests, args.overlay_requests, trace_manifest
    )
    expected_qwen = ModelManifest.from_json(load_object(args.qwen_manifest))
    expected_gemma = ModelManifest.from_json(load_object(args.gemma_manifest))
    llama_model_id = overlay_rows[0]["execution_model_id"]
    model_paths = {
        expected_qwen.model_id: args.qwen_model,
        expected_gemma.model_id: args.gemma_model,
        llama_model_id: args.llama_model,
    }
    catalog = _apply_phone_thermal_status_limits(
        RuntimeCapabilityCatalog.from_json(load_object(args.capability_catalog)),
        getattr(args, "phone_thermal_status_limits_json", None),
    )
    multi_session_catalog = any(
        len(executor.phone_sessions) > 1
        for executor in catalog.executors
    )
    require(
        not multi_session_catalog
        or (
            type(args.phone_resident_workers) is str
            and bool(args.phone_resident_workers)
            and type(args.phone_resident_router) is str
            and bool(args.phone_resident_router)
            and type(args.phone_multi_session_port_base) is int
            and 0 < args.phone_multi_session_port_base <= 65535
        ),
        "multi-session phone physical binding",
    )
    diagnostic_url = urlsplit(args.phone_diagnostic_endpoint)
    require(
        diagnostic_url.scheme == "http"
        and diagnostic_url.hostname is not None
        and diagnostic_url.port is not None,
        "phone diagnostic endpoint",
    )
    return _TraceModels(
        trace_manifest=trace_manifest,
        large_rows=large_rows,
        overlay_rows=overlay_rows,
        expected_qwen=expected_qwen,
        expected_gemma=expected_gemma,
        llama_model_id=llama_model_id,
        model_paths=model_paths,
        catalog=catalog,
        diagnostic_url=diagnostic_url,
    )


def _configured_adaptive_decode_config(
    args: argparse.Namespace,
) -> AdaptiveDecodeConfig | None:
    overrides = _adaptive_decode_overrides_from_json(
        getattr(args, "adaptive_decode_overrides_json", None)
    )
    overrides.update({
        name: getattr(args, "adaptive_" + name)
        for name in (
            "minimum_remaining_tokens",
            "maximum_probe_attempts_per_context",
        )
        if getattr(args, "adaptive_" + name) is not None
    })
    if not overrides:
        return None
    try:
        return AdaptiveDecodeConfig(**overrides)
    except TypeError as exc:
        raise UnifiedTraceError("adaptive decode override names are invalid: " + str(exc)) from exc


def _adaptive_decode_overrides_from_json(text: str | None) -> dict[str, object]:
    """Campaign `adaptive_decode_overrides` arrive as one JSON object; lists become tuples."""
    if text is None:
        return {}
    value = json.loads(text)
    require(type(value) is dict and value, "adaptive decode overrides must be a non-empty object")
    return {
        name: tuple(item) if type(item) is list else item
        for name, item in value.items()
    }


def _dispatch_policy_from_json(text: str | None) -> RuntimeDispatchPolicy | None:
    """Campaign `dispatch_policy` arrives as one JSON object; absent keeps the legacy order."""
    if text is None:
        return None
    try:
        return RuntimeDispatchPolicy.from_json(json.loads(text))
    except (TypeError, ValueError) as exc:
        raise UnifiedTraceError("dispatch policy is invalid: " + str(exc)) from exc


def _require_continuous_join_prerequisites(
    policy: RuntimeDispatchPolicy, adaptive_config: AdaptiveDecodeConfig | None,
) -> None:
    """A joiner gets phone assistance only through server policy coherence; refuse to run without it."""
    require(
        not policy.continuous_join
        or (adaptive_config is not None and adaptive_config.server_policy_coherence),
        "dispatch policy continuous_join requires adaptive_decode_overrides.server_policy_coherence",
    )


def _build_scheduler(
    args: argparse.Namespace,
    models: _TraceModels,
    *,
    adaptive_decode_config: AdaptiveDecodeConfig | None = None,
    load_adaptive_observations: bool = True,
) -> tuple[UnifiedScheduler, dict[str, ModelManifest], frozenset[str]]:
    adaptive_config = (
        adaptive_decode_config
        if adaptive_decode_config is not None
        else _configured_adaptive_decode_config(args)
    )
    scheduler = UnifiedScheduler.for_runtime_discovery(
        "enforce",
        adaptive_decode_config=adaptive_config,
        maximum_phone_sessions=args.maximum_phone_sessions,
        protected_work_policy=args.protected_work_policy,
    )
    dispatch_policy = _dispatch_policy_from_json(getattr(args, "dispatch_policy_json", None))
    if dispatch_policy is not None:
        _require_continuous_join_prerequisites(dispatch_policy, adaptive_config)
        scheduler.configure_runtime_dispatch_policy(dispatch_policy)
    scheduler.register_runtime_capabilities(models.catalog)
    manifests = {
        model_id: scheduler.register_gguf_model(
            model_id,
            path,
            cache_path=args.gguf_manifest_cache,
        )
        for model_id, path in models.model_paths.items()
    }
    for expected in (models.expected_qwen, models.expected_gemma):
        actual = manifests[expected.model_id]
        require(
            actual.artifact_sha256 == expected.artifact_sha256
            and actual.artifact_bytes == expected.artifact_bytes
            and actual.block_count == expected.block_count
            and actual.embedding_length == expected.embedding_length
            and actual.feed_forward_length == expected.feed_forward_length,
            f"GGUF manifest identity: {expected.model_id}",
        )
    shard_indexes = _ffn_shard_indexes({
        manifests[models.expected_qwen.model_id].artifact_sha256:
            args.qwen_ffn_shards,
        manifests[models.expected_gemma.model_id].artifact_sha256:
            args.gemma_ffn_shards,
        manifests[models.llama_model_id].artifact_sha256:
            args.llama_ffn_shards,
    })
    if shard_indexes:
        scheduler.register_phone_ffn_shard_storage(_ffn_shard_storage(shard_indexes))
    if args.observation_store_input is not None:
        source_catalog = (
            None
            if args.observation_source_catalog is None
            else RuntimeCapabilityCatalog.from_json(load_object(
                args.observation_source_catalog
            ))
        )
        scheduler.load_automated_observations(
            load_object(args.observation_store_input),
            source_catalog=source_catalog,
        )
    if (
        load_adaptive_observations
        and args.adaptive_observation_store_input is not None
    ):
        source_catalog = (
            None
            if args.adaptive_observation_source_catalog is None
            else RuntimeCapabilityCatalog.from_json(load_object(
                args.adaptive_observation_source_catalog
            ))
        )
        scheduler.load_adaptive_decode_observations(
            load_object(args.adaptive_observation_store_input),
            source_catalog=source_catalog,
        )
    initial_adaptive_group_sha256s = frozenset(
        group["grouped_observation_sha256"]
        for group in scheduler.adaptive_decode_observation_snapshot().get(
            "groups", []
        )
    )
    fixed = getattr(args, "fixed_phone_residency_json", None)
    if fixed is not None:
        scheduler.configure_fixed_phone_residency(
            FixedPhoneResidencyConfiguration.from_json(json.loads(fixed))
        )
    caps = getattr(args, "phone_htp_memory_caps_json", None)
    if caps is not None:
        rows = json.loads(caps)
        require(isinstance(rows, list), "phone HTP memory caps must be an array")
        configured = tuple(PhoneHtpMemoryCapConfiguration.from_json(row) for row in rows)
        require(len({cap.phone_device_id for cap in configured}) == len(configured),
                "phone HTP memory cap devices are not unique")
        for cap in sorted(configured, key=lambda row: row.phone_device_id):
            scheduler.set_phone_htp_memory_cap(
                cap.phone_device_id, cap.cap_bytes, workspace_bytes=cap.workspace_bytes,
                observed_at_us=0,
            )
    reprovisioning = getattr(args, "phone_resident_model_reprovisioning_json", None)
    if reprovisioning is not None:
        scheduler.configure_phone_resident_model_reprovisioning(
            PhoneResidentModelReprovisioningConfiguration.from_json(json.loads(reprovisioning))
        )
    return scheduler, manifests, initial_adaptive_group_sha256s


def _select_replay(
    args: argparse.Namespace,
    models: _TraceModels,
    manifests: dict[str, ModelManifest],
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any], bool]:
    inventory = models.trace_manifest["model_inventory"]
    for model_id, manifest in manifests.items():
        expected = inventory[model_id]
        require(
            manifest.artifact_sha256
                == "sha256:" + expected["artifact_sha256"]
            and manifest.artifact_bytes == expected["artifact_bytes"],
            f"trace model identity: {model_id}",
        )
    aliases = artifact_aliases(models.catalog, manifests)
    merged = merge_rows(models.large_rows, models.overlay_rows, {
        QWEN_ROLE: models.expected_qwen.model_id,
        GEMMA_ROLE: models.expected_gemma.model_id,
    })
    if args.replay_schedule is None:
        merged = select_rows(merged, args.request_indices)
        merged, replay_schedule = scale_replay_arrivals(
            merged, args.arrival_scale
        )
    else:
        merged, replay_schedule = apply_named_replay_schedule(
            merged, load_object(args.replay_schedule)
        )
    isolated_cohort_expected = (
        args.energy_attribution_kind == "isolated" and len(merged) > 1
    )
    require(
        not isolated_cohort_expected
        or (
            args.selection_mode == "adaptive-decode"
            and 2 <= len(merged) <= 4
        ),
        "multi-request isolated energy requires one adaptive cohort",
    )
    return aliases, merged, replay_schedule, isolated_cohort_expected


def _direct_phone_configuration(
    args: argparse.Namespace,
    models: _TraceModels,
    manifests: dict[str, ModelManifest],
    host_dependencies: dict[str, Path],
    transport_identity,
) -> DirectPhoneFfnSessionConfiguration:
    expected_qwen = models.expected_qwen
    expected_gemma = models.expected_gemma
    diagnostic_url = models.diagnostic_url
    phone_models = {
        manifests[expected_qwen.model_id].artifact_sha256: args.qwen_phone_model,
        manifests[expected_gemma.model_id].artifact_sha256: args.gemma_phone_model,
    }
    if args.llama_ffn_shards is not None:
        require(bool(args.llama_phone_model), "overlay phone parent artifact path")
        phone_models[manifests[models.llama_model_id].artifact_sha256] = args.llama_phone_model
    return DirectPhoneFfnSessionConfiguration(
        adb_path=args.adb,
        usb_close_path=args.phone_usb_close,
        serial=args.phone_usb_serial,
        adb_port=args.adb_port,
        session_script=args.phone_session,
        restore_script=args.phone_restore,
        session_root=args.phone_session_root,
        worker_paths_by_artifact={
            artifact: args.phone_worker for artifact in phone_models
        },
        model_paths_by_artifact=phone_models,
        ffn_shards_by_artifact=_ffn_shard_indexes({
            manifests[expected_qwen.model_id].artifact_sha256:
                args.qwen_ffn_shards,
            manifests[expected_gemma.model_id].artifact_sha256:
                args.gemma_ffn_shards,
            **({manifests[models.llama_model_id].artifact_sha256: args.llama_ffn_shards}
               if args.llama_ffn_shards is not None else {}),
        }),
        backend_by_device={"op15-phone": "HTP0"},
        minimum_usb_speed_mbps=args.minimum_usb_speed_mbps,
        required_kernel_release=args.phone_kernel_release,
        diagnostic_port=diagnostic_url.port,
        diagnostic_host=diagnostic_url.hostname,
        busybox_path=args.phone_busybox,
        network_manager_path=args.nmcli,
        transport_qualification_identity=transport_identity,
        transport_host_binary_path=(
            args.server if transport_identity is not None else None
        ),
        transport_host_dependency_paths=(
            host_dependencies if transport_identity is not None else {}
        ),
        phone_boot_image_sha256=(
            args.phone_boot_image_sha256
            if transport_identity is not None else None
        ),
        android_gadget_path=args.phone_android_gadget,
        functionfs_gadget_path=args.phone_functionfs_gadget,
        functionfs_root_path=args.phone_functionfs_root,
        phone_usb_controller=args.phone_usb_controller,
        remote_hash_cache_path=args.phone_remote_hash_cache,
        resident_workers_path=args.phone_resident_workers,
        resident_router_path=args.phone_resident_router,
        cpu_affinity=getattr(args, "phone_cpu_affinity", None),
        multi_session_port_base=args.phone_multi_session_port_base,
        multi_session_device_count=max(
            (
                len(executor.phone_sessions)
                for executor in models.catalog.executors
            ),
            default=0,
        ) or None,
    )


def _ffn_shard_indexes(
    specs: Mapping[str, str | None],
) -> dict[str, FfnShardIndex]:
    """Parse ``LOCAL_FFN_SHARDS.json=PHONE_DIR`` per artifact; verify parents."""
    indexes: dict[str, FfnShardIndex] = {}
    for artifact, spec in specs.items():
        if spec is None:
            continue
        local, separator, remote_dir = spec.rpartition("=")
        require(bool(separator) and bool(local), "ffn shard spec must be LOCAL=PHONE_DIR")
        try:
            index = FfnShardIndex.load(Path(local), remote_dir)
        except FfnShardIndexError as error:
            raise UnifiedTraceError(str(error)) from error
        require(
            index.parent_sha256 == artifact,
            "ffn shard index parent differs from the registered phone model",
        )
        indexes[artifact] = index
    return indexes


def _ffn_shard_storage(indexes: Mapping[str, FfnShardIndex]) -> tuple[PhoneFfnShardStorageMetadata, ...]:
    return tuple(
        PhoneFfnShardStorageMetadata(
            parent_artifact_sha256=index.parent_sha256,
            shard_sha256=record.shard_sha256,
            path=record.remote_path,
            layer_mask=record.layer_mask,
            maximum_columns=record.columns,
            session_id=record.session_hint,
        )
        for index in indexes.values()
        for record in index.records
    )


def _android_phone_server(
    args: argparse.Namespace,
    models: _TraceModels,
    manifests: dict[str, ModelManifest],
) -> AndroidLlamaServerProcessConfiguration | None:
    whole_phone_values = (
        args.phone_whole_server,
        args.phone_whole_library_directory,
        args.phone_whole_model,
        args.phone_whole_state_directory,
    )
    require(
        all(value is None for value in whole_phone_values)
        or all(type(value) is str and value for value in whole_phone_values),
        "physical whole-phone configuration",
    )
    return (
        None
        if args.phone_whole_server is None
        else AndroidLlamaServerProcessConfiguration(
            adb_path=args.adb,
            serial=args.phone_usb_serial,
            adb_port=args.adb_port,
            remote_server_path=args.phone_whole_server,
            remote_library_directory=(
                args.phone_whole_library_directory
            ),
            remote_model_paths_by_artifact={
                manifests[models.llama_model_id].artifact_sha256:
                    args.phone_whole_model,
            },
            remote_state_directory=args.phone_whole_state_directory,
            executable_device_name=args.phone_whole_executable_device,
            control_transport=args.phone_whole_control_transport,
            ncm_adb_endpoint=args.phone_whole_ncm_adb_endpoint,
            output_directory=args.output,
        )
    )


def _build_rig(
    args: argparse.Namespace,
    models: _TraceModels,
    manifests: dict[str, ModelManifest],
    host_dependencies: dict[str, Path],
) -> HeterogeneousPhysicalRig:
    from research_dev.scheduler.campaigns.burstgpt.helper_phone_evidence import campaign_co_helper_lifecycles
    co_helpers = campaign_co_helper_lifecycles(getattr(args, "helper_phone_evidence", ()),
                                             models.catalog, manifests, args.server)
    transport_identity = (
        None
        if args.usb_qualification_identity is None
        else load_transport_qualification_identity(
            args.usb_qualification_identity
        )
    )
    direct_phone_configuration = _direct_phone_configuration(
        args, models, manifests, host_dependencies, transport_identity
    )
    android_phone_server = _android_phone_server(args, models, manifests)
    device_power = getattr(args, "device_power", None)
    return HeterogeneousPhysicalRig(
        HeterogeneousRigConfiguration(
            catalog=models.catalog,
            manifests=manifests,
            server_path=args.server,
            resident_server_path=args.resident_server,
            model_paths_by_artifact={
                manifest.artifact_sha256: models.model_paths[model_id]
                for model_id, manifest in manifests.items()
            },
            cuda_library_directory=args.cuda_lib_dir,
            resident_library_directory=args.resident_lib_dir,
            bridge_path=args.bridge,
            close_helper_path=args.close_helper,
            direct_phone_session=direct_phone_configuration,
            host_metrics=default_host_metric_callbacks(gpu_clocks=device_power is not None),
            phone_diagnostic_endpoint=args.phone_diagnostic_endpoint,
            phone_usb_serial=args.phone_usb_serial,
            phone_device_id="op15-phone",
            elastic_phones=getattr(args, "elastic_phones", None),
            device_power=device_power,
            speculative_rows=getattr(args, "speculative_rows", None),
            phone_memory_resource_id="op15-ram",
            gpu_device_id="desktop-cuda",
            gpu_memory_resource_id="cuda0-vram",
            host_memory_resource_id="host-ram",
            adb_port=args.adb_port,
            minimum_usb_speed_mbps=args.minimum_usb_speed_mbps,
            output_directory=args.output,
            large_phase_id_by_model={
                models.expected_qwen.model_id: 1,
                models.expected_gemma.model_id: 3,
            },
            transition_phase_id=2,
            preloaded_model_by_executor={
                "physical:desktop-cpu": models.llama_model_id,
            },
            active_device_cost_features={
                "large_model_op15": "op15-phone",
            },
            resident_model_id=models.llama_model_id,
            resident_executor_id="physical:desktop-cpu",
            android_phone_server=android_phone_server,
            energy_attribution_kind=args.energy_attribution_kind,
            host_memory_budget_bytes=getattr(args, "host_memory_budget_bytes", None),
            co_helper_lifecycles=co_helpers,
        ),
        epoch_ns=0,
    )


def _warm_payload(
    models: _TraceModels, aliases: dict[str, str], streams: Path
) -> LlamaCppCompletionPayload:
    first_overlay = models.overlay_rows[0]
    return LlamaCppCompletionPayload(
        request_id="unified-resident-warm",
        expected_model_alias=aliases[models.llama_model_id],
        input_tokens=first_overlay["input_tokens"],
        output_tokens=2,
        prompt_tokens=tuple(first_overlay["prompt_tokens"]),
        seed=2_147_000_000,
        stream_path=streams / "warm-template.raw",
        on_first_token=lambda _value: None,
        quality_mode="accounting-only",
    )


def _preload_startup_parents(args, scheduler, rig, aliases, streams, epoch_ns, configurations):
    """Invoke exact-parent startup verification and persist its paid receipts."""
    results = []
    for configuration in configurations:
        at_us = (time.monotonic_ns() - epoch_ns) // 1000
        request_id = "startup-parent:" + configuration.executor_id
        request = Request(
            request_id=request_id, workload_id="startup:" + configuration.model_id,
            arrival_us=at_us, deadline_us=at_us + 300_000_000,
            input_tokens=1, output_tokens=2, quality_requirement="exact",
        )

        def snapshot(ticket, observed_at_us):
            value = rig.snapshot(ticket.request, configuration.model_id, observed_at_us)
            path = args.output / "snapshots" / f"startup-{len(results)}-{observed_at_us}.json"
            with path.open("xb") as stream:
                stream.write(canonical(value.to_json()))
            return value

        initial = rig.snapshot(request, configuration.model_id, at_us)
        ticket = scheduler.submit_startup_parent_preload(
            request, configuration.model_id, initial,
            executor_id=configuration.executor_id,
            desktop_placement_sha256=configuration.desktop_placement_sha256,
            observed_at_us=max(at_us, initial.captured_at_us),
        )
        with (args.output / f"STARTUP_TICKET_{len(results)}.json").open("xb") as stream:
            stream.write(canonical({"configuration": configuration.to_json(),
                                    "snapshot": initial.to_json(), "ticket": ticket.to_json()}))
        adapter = CanonicalPhysicalAdapter(
            scheduler, rig.backend(), epoch_ns=epoch_ns, snapshot_provider=snapshot,
        )
        execution = adapter.execute(ticket, LlamaCppCompletionPayload(
            request_id=request_id, expected_model_alias=aliases[configuration.model_id],
            input_tokens=1, output_tokens=2, prompt_tokens=(1,), seed=42,
            stream_path=streams / f"startup-parent-{len(results)}.raw",
            on_first_token=lambda _value: None, quality_mode="accounting-only",
        ))
        ready = dict(rig.verify_startup_parent(execution.command))
        require(ready["desktop_placement_sha256"] == configuration.desktop_placement_sha256,
                "startup verified desktop placement differs")
        terminal = scheduler.runtime_ticket(request_id)
        results.append({
            "request_id": request_id, "configuration": configuration.to_json(),
            "started_at_us": at_us, "ready": ready,
            "command": execution.command.to_json(), "terminal_ticket": terminal.to_json(),
            "verification_energy": measured_energy(execution.observation.energy),
            "execution_proof": execution.observation.payload.get("physical_execution_proof"),
            "accounting": "startup-load-and-verification-in-paid-interval",
        })
        with (args.output / f"STARTUP_PRELOAD_{len(results) - 1}.json").open("xb") as stream:
            stream.write(canonical(results[-1]))
    return results


def _preload_fixed_residency(args, scheduler, rig, aliases, streams, epoch_ns):
    """Invoke scheduler-owned assignment and transaction paths; persist evidence."""
    requests = scheduler.fixed_phone_residency_requests()
    model_id, shapes = next(iter(requests.items()))
    sequence = 0

    def capture(request, model, at_us):
        nonlocal sequence
        snapshot = rig.snapshot(request, model, at_us)
        path = args.output / "snapshots" / f"fixed-preparation-{sequence:04d}.json"
        sequence += 1
        with path.open("xb") as stream:
            stream.write(canonical(snapshot.to_json()))
        return snapshot

    def now_us():
        return (time.monotonic_ns() - epoch_ns) // 1000

    def payload(stage):
        return LlamaCppCompletionPayload(
            request_id=stage.request.request_id,
            expected_model_alias=aliases[stage.model_id], input_tokens=1,
            output_tokens=2, prompt_tokens=(1,), seed=42,
            stream_path=streams / (stage.stage_id + ".raw"),
            on_first_token=lambda _value: None, quality_mode="accounting-only",
        )

    started_ns = time.monotonic_ns()
    initial = capture(shapes[0], model_id, now_us())
    plan = CanonicalOfflinePhoneResidencyPreloader.plan_with_observation_refresh(
        scheduler, requests, snapshot=initial,
        snapshot_provider=lambda at_us: capture(shapes[0], model_id, at_us),
        refresh_observation=rig.request_runtime_observation_refresh,
        epoch_ns=epoch_ns,
    )
    preloader = CanonicalOfflinePhoneResidencyPreloader(
        scheduler, rig.backend(), epoch_ns=epoch_ns,
        snapshot_provider=lambda stage, at_us: capture(stage.request, stage.model_id, at_us),
    )
    result = preloader.preload(
        plan, payload, initial_snapshot=capture(shapes[0], model_id, now_us()),
        observed_at_us=now_us(),
    )
    finished_ns = time.monotonic_ns()
    require(result.plan.state == "READY", "fixed residency preparation did not reach READY")
    with (args.output / "FIXED_RESIDENCY_PREPARATION.json").open("xb") as stream:
        stream.write(canonical({
            **result.to_json(), "started_ns": started_ns, "finished_ns": finished_ns,
            "duration_us": (finished_ns - started_ns) // 1000,
            "energy_interval_note": "May overlap request execution; included once in the complete paid interval.",
            "energy": measured_energy(rig.trace_energy(started_ns, finished_ns)),
        }))
    return result


def _snapshot_provider(
    rig: HeterogeneousPhysicalRig,
    scheduler: UnifiedScheduler,
    snapshots: Path,
):
    runtime_snapshot_lock = threading.Lock()
    runtime_snapshot_sequence = 0

    def persist_runtime_snapshot(ticket, observed_at_us):
        nonlocal runtime_snapshot_sequence
        snapshot = rig.snapshot_for_ticket(ticket, observed_at_us)
        with runtime_snapshot_lock:
            sequence = runtime_snapshot_sequence
            runtime_snapshot_sequence += 1
        path = snapshots / (
            "runtime-"
            + f"{sequence:03d}-"
            + ticket.request.request_id.replace(":", "-")
            + "-"
            + str(observed_at_us)
            + ".json"
        )
        path.write_bytes(canonical(snapshot.to_json()))
        current = scheduler.runtime_ticket(ticket.request.request_id)
        (path.with_suffix(".scheduler.json")).write_bytes(canonical({
            "observed_at_us": observed_at_us,
            "request_id": ticket.request.request_id,
            "runtime_controller": dict(
                scheduler.runtime_controller_snapshot()
            ),
            "ticket": {
                "attempt_index": current.attempt_index,
                "dispatch_state": current.dispatch_state,
                "finish_upper_us": current.decision.finish_upper_us,
                "lease_status": current.lease_status,
                "route_id": current.decision.route_id,
                "start_us": current.decision.start_us,
                "ticket_id": current.ticket_id,
                "transition_status": current.transition_status,
            },
        }))
        return snapshot

    return persist_runtime_snapshot


@dataclass
class _ArrivalState:
    first_token_ns: dict[str, int] = field(default_factory=dict)
    # every attempt's first token (a recovered request streams more than once)
    first_token_history: dict[str, list[int]] = field(default_factory=dict)
    epoch_ns: int | None = None
    first_token_lock: threading.Lock = field(default_factory=threading.Lock)
    active_slot_samples: list[dict[str, object]] = field(
        default_factory=list
    )
    active_slot_lock: threading.Lock = field(default_factory=threading.Lock)
    overheads: dict[str, dict[str, int]] = field(default_factory=dict)
    rejections: dict[str, dict[str, Any]] = field(default_factory=dict)


def _submit_arrivals(
    args: argparse.Namespace,
    merged: list[dict[str, Any]],
    coordinator: CanonicalArrivalCoordinator,
    rig: HeterogeneousPhysicalRig,
    epoch_ns: int,
    streams: Path,
    snapshots: Path,
    aliases: dict[str, str],
    state: _ArrivalState,
) -> None:
    state.epoch_ns = epoch_ns
    for item in merged:
        combined_index = item["combined_index"]
        row = item["row"]
        model_id = item["model_id"]
        request = Request(
            request_id=row["event_id"],
            workload_id="physical:" + model_id,
            arrival_us=row["arrival_us"],
            deadline_us=row["arrival_us"] + row["slo_us"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            quality_requirement="semantic",
        )
        _note_next_arrival(rig, request.arrival_us)
        coordinator.wait_for_arrival(request.arrival_us)
        snapshot_started_ns = time.monotonic_ns()
        captured_at_us = coordinator.observed_at_us()
        snapshot = rig.snapshot(request, model_id, captured_at_us)
        snapshot_finished_ns = time.monotonic_ns()
        (snapshots / f"request-{combined_index:03d}.json").write_bytes(
            canonical(snapshot.to_json())
        )

        def on_first(value_ns: int, request_id=request.request_id) -> None:
            with state.first_token_lock:
                state.first_token_ns.setdefault(request_id, value_ns)
                state.first_token_history.setdefault(request_id, []).append(value_ns)

        def on_active_batch(
            value: int,
            request_id=request.request_id,
            sampled_model_id=model_id,
        ) -> None:
            with state.active_slot_lock:
                state.active_slot_samples.append({
                    "active_slots": value,
                    "model_id": sampled_model_id,
                    "observed_at_us": max(
                        0,
                        (time.monotonic_ns() - epoch_ns) // 1000,
                    ),
                    "request_id": request_id,
                })

        payload = LlamaCppCompletionPayload(
            request_id=request.request_id,
            expected_model_alias=aliases[model_id],
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
            prompt_tokens=tuple(row["prompt_tokens"]),
            seed=combined_index,
            stream_path=streams / f"request-{combined_index:03d}.raw",
            on_first_token=on_first,
            on_active_batch=on_active_batch,
            quality_mode="semantic",
        )
        submit_started_ns = time.monotonic_ns()
        ticket = coordinator.submit(
            CanonicalRuntimeSubmission(
                request=request,
                model_id=model_id,
                snapshot=snapshot,
                payload=payload,
                selection_mode=args.selection_mode,
            ),
            observed_at_us=coordinator.observed_at_us(),
        )
        submit_finished_ns = time.monotonic_ns()
        state.overheads[request.request_id] = {
            "scheduler_submit_ns": submit_finished_ns - submit_started_ns,
            "snapshot_capture_ns": (
                snapshot_finished_ns - snapshot_started_ns
            ),
        }
        if ticket is None:
            # Permanently unsupported shape: recorded with its exact reason,
            # the campaign continues with every other request.
            rejection = coordinator.rejections()[-1]
            require(
                rejection.request_id == request.request_id,
                "submission rejection identity",
            )
            state.rejections[request.request_id] = {
                **rejection.to_json(),
                "combined_request_index": combined_index,
                "input_tokens": request.input_tokens,
                "output_tokens": request.output_tokens,
            }
            print(
                "REJECTED " + request.request_id + " (" + rejection.reason
                + ") index " + str(combined_index),
                flush=True,
            )
    _note_next_arrival(rig, None)


def _screen_request_shapes(
    scheduler: UnifiedScheduler, merged: list[dict[str, Any]]
) -> dict[str, Any]:
    verdicts = []
    for item in merged:
        row = item["row"]
        request = Request(
            request_id=row["event_id"],
            workload_id="physical:" + item["model_id"],
            arrival_us=row["arrival_us"],
            deadline_us=row["arrival_us"] + row["slo_us"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            quality_requirement="semantic",
        )
        verdict = scheduler.request_shape_support(
            request, item["model_id"]
        ).to_json()
        verdict["combined_request_index"] = item["combined_index"]
        verdicts.append(verdict)
    return {
        "checked": len(verdicts),
        "unsupported": [row for row in verdicts if not row["supported"]],
        "verdicts": verdicts,
    }


@dataclass(frozen=True)
class _JournalSummary:
    journal: dict[str, Any]
    records: list[dict[str, Any]]
    attempts: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    terminals: list[dict[str, Any]]
    candidate_coverage: Any


def _summarize_journal(
    args: argparse.Namespace,
    scheduler: UnifiedScheduler,
    catalog: RuntimeCapabilityCatalog,
    manifests: dict[str, ModelManifest],
    merged: list[dict[str, Any]],
    startup_request_ids: tuple[str, ...] = (),
    rejected_request_ids: frozenset[str] = frozenset(),
) -> _JournalSummary:
    journal = scheduler.runtime_decision_log()
    scheduler.validate_runtime_decision_log(journal)
    startup_ids = set(startup_request_ids)
    # Requests rejected at submission never received a ticket and are
    # accounted for in RESULT.rejected_requests, not in the journal.
    merged = [
        item for item in merged
        if item["row"]["event_id"] not in rejected_request_ids
    ]
    trace_ids = {item["row"]["event_id"] for item in merged}
    require(not trace_ids & startup_ids, "startup and trace request identities overlap")
    require(all(set(row["request_ids"]) <= trace_ids | startup_ids
                for row in journal["records"]), "journal contains an unaccounted request")
    records = [row for row in journal["records"] if not set(row["request_ids"]) & startup_ids]
    attempts = [
        row for row in records
        if row["event_kind"] in {"DECISION", "REPLAN", "FALLBACK"}
    ]
    decisions = [
        row for row in attempts
        if row["event_kind"] == "DECISION"
        and row["attempt_index"] == 0
    ]
    terminals = [
        row for row in records
        if row["event_kind"] in {"COMPLETED", "FAILED", "CANCELLED"}
    ]
    expected_requests = len(merged)
    if (
        args.request_indices is None
        and args.replay_schedule is None
        and not rejected_request_ids
    ):
        require(
            len(decisions) == 84
            and len(terminals) == 84
            and len({row["request_ids"][0] for row in decisions}) == 84
            and len({row["request_ids"][0] for row in terminals}) == 84,
            "scheduler journal request coverage",
        )
    else:
        require(
            len(decisions) == expected_requests
            and len(terminals) == expected_requests
            and len({
                row["request_ids"][0] for row in decisions
            }) == expected_requests
            and len({
                row["request_ids"][0] for row in terminals
            }) == expected_requests,
            "scheduler journal request coverage",
        )
    validate_attempt_chains(
        attempts, {item["row"]["event_id"] for item in merged}
    )
    manifest_by_request_id = {
        item["row"]["event_id"]: manifests[item["model_id"]]
        for item in merged
    }
    candidate_coverage = validate_decision_candidate_coverage(
        catalog, records, manifest_by_request_id
    )
    require(
        len(candidate_coverage) == expected_requests,
        "per-request candidate coverage",
    )
    return _JournalSummary(
        journal=journal,
        records=records,
        attempts=attempts,
        decisions=decisions,
        terminals=terminals,
        candidate_coverage=candidate_coverage,
    )


def _recovered_request_timing(
    observation,
    recovered: list[dict[str, Any]],
    first_tokens: Sequence[int],
    epoch_ns: int | None,
) -> tuple[int, int | None, list[dict[str, Any]]]:
    """Latency, first token and recovery rows of a recovered request (elastic phones).

    The latency runs from the first attempt's start to the terminal finish (it includes the failed
    attempts, the failure classification wait and any reload); the first token is the terminal
    attempt's. Each REQUEST_RECOVERED row keeps the per-attempt numbers: ``attempt_latency_us`` (the
    terminal attempt alone, what a non-recovered row reports) and ``first_attempt_ttft_us`` (the
    discarded first attempt's first token after its own start; None when it streamed none).
    """
    starts = [row.get("attempt_started_us") for row in recovered]
    require(
        type(epoch_ns) is int
        and all(type(value) is int and 0 <= value <= observation.started_us for value in starts),
        "recovered request attempt timing",
    )
    starts.sort()
    terminal_start_ns = epoch_ns + observation.started_us * 1_000
    second_start_ns = epoch_ns + ([*starts[1:], observation.started_us][0]) * 1_000
    terminal_first = next((value for value in first_tokens if value >= terminal_start_ns), None)
    first_attempt_first = next((
        value for value in first_tokens
        if epoch_ns + starts[0] * 1_000 <= value < second_start_ns
    ), None)
    per_attempt = {
        "attempt_latency_us": observation.finished_us - observation.started_us,
        "first_attempt_ttft_us": (
            None if first_attempt_first is None
            else max(0, (first_attempt_first - epoch_ns) // 1_000 - starts[0])
        ),
    }
    return (
        observation.finished_us - starts[0],
        terminal_first,
        [{**row, **per_attempt} for row in recovered],
    )


def _request_result(
    scheduler: UnifiedScheduler,
    completed,
    request_id: str,
    combined_index: int,
    item: dict[str, Any],
    current_adaptive_groups: list[dict[str, Any]],
    state: _ArrivalState,
) -> dict[str, Any]:
    execution = completed.executions[request_id]
    terminal_ticket = scheduler.runtime_ticket(request_id)
    row = item["row"]
    require(
        execution.command.endpoint
            == execution.ticket.binding.endpoint
        and execution.command.executor_id
            == execution.ticket.binding.executor_id
        and execution.command.operator_plan_sha256
            == execution.ticket.execution_plan.plan_sha256
        and terminal_ticket.execution_receipt is not None
        and terminal_ticket.execution_receipt.endpoint
            == execution.command.endpoint,
        "physical execution differs from scheduler ticket",
    )
    execution_plan = execution.ticket.execution_plan
    execution_contract = (
        None
        if execution_plan is None
        else execution_plan.execution_contract
    )
    physical_execution_proof = (
        execution.observation.payload.get(
            "physical_execution_proof"
        )
    )
    phone_call_count = (
        0
        if type(physical_execution_proof) is not dict
        else int(
            physical_execution_proof.get("phone_call_count") or 0
        )
    )
    fraction_history = execution_fraction_history(
        execution_contract,
        execution.ticket.ticket_id,
        current_adaptive_groups,
        phone_call_count,
        request_id=request_id,
        decode_cohort=execution.ticket.decode_cohort,
    )
    # elastic phones (drop recovery): REQUEST_RECOVERED rows extend the
    # existing attempt_ticket_ids / recoveries; absent for every other run.
    recovered = [
        dict(row) for row in getattr(execution, "recovery_events", ())
    ]
    latency_us = (
        execution.observation.finished_us
        - execution.observation.started_us
    )
    first_token_ns = state.first_token_ns.get(request_id)
    if recovered:
        latency_us, first_token_ns, recovered = _recovered_request_timing(
            execution.observation,
            recovered,
            state.first_token_history.get(request_id, ()),
            state.epoch_ns,
        )
    # speculative rows: the server's draft statistics of this request; only a request whose
    # completion carried them (campaign speculative_rows) has the key
    speculative = execution.observation.payload.get("speculative")
    return {
        **({"request_recovered": recovered} if recovered else {}),
        **({"speculative": dict(speculative)} if type(speculative) is dict else {}),
        "actual_endpoint": execution.command.endpoint,
        "actual_executor_id": execution.command.executor_id,
        "actual_latency_us": latency_us,
        "attempt_ticket_ids": list(execution.attempt_ticket_ids),
        "combined_request_index": combined_index,
        "completion": execution.completion.to_json(),
        "dispatch_receipts": [
            receipt.to_json()
            for receipt in execution.dispatch_receipts
        ],
        "execution_command": execution.command.to_json(),
        "first_token_ns": first_token_ns,
        "fraction_history": fraction_history,
        "initial_ticket": completed.tickets[request_id].to_json(),
        "measured_energy": measured_energy(
            execution.observation.energy
        ),
        "measured_energy_scope": (
            "overlapping_request_window_calibration_only"
        ),
        "model_id": item["model_id"],
        "input_tokens": terminal_ticket.request.input_tokens,
        "output_tokens": terminal_ticket.request.output_tokens,
        "prompt_sha256": "sha256:" + hashlib.sha256(
            canonical(row["prompt_tokens"])
        ).hexdigest(),
        "seed": combined_index,
        "output_sha256": execution.observation.output_sha256,
        "output_quality": execution.observation.payload.get(
            "output_quality"
        ),
        "physical_execution_proof": physical_execution_proof,
        "recoveries": [
            recovery.to_json() for recovery in execution.recoveries
        ],
        "request_id": request_id,
        "scheduling_overhead": state.overheads[request_id],
        "source": item["source"],
        "source_arrival_us": row["source_arrival_us"],
        "source_slo_us": row["slo_us"],
        "replay_arrival_us": row["replay_arrival_us"],
        "terminal_ticket": terminal_ticket.to_json(),
        "trace_arrival_us": row["source_arrival_us"],
    }


def _build_request_results(
    scheduler: UnifiedScheduler,
    completed,
    merged: list[dict[str, Any]],
    current_adaptive_groups: list[dict[str, Any]],
    state: _ArrivalState,
) -> list[dict[str, Any]]:
    request_results = []
    by_request = {
        item["row"]["event_id"]: (item["combined_index"], item)
        for item in merged
    }
    for request_id in completed.request_ids:
        combined_index, item = by_request[request_id]
        request_results.append(_request_result(
            scheduler,
            completed,
            request_id,
            combined_index,
            item,
            current_adaptive_groups,
            state,
        ))
    return request_results


def _require_isolated_cohort(
    decode_cohort_state: dict[str, Any],
    merged: list[dict[str, Any]],
    request_results: list[dict[str, Any]],
) -> None:
    receipts = decode_cohort_state["receipts"]
    require(
        len(receipts) == 1,
        "isolated cohort requires exactly one energy receipt",
    )
    cohort_id, cohort_receipt = next(iter(receipts.items()))
    request_ids = {item["row"]["event_id"] for item in merged}
    require(
        set(cohort_receipt["member_request_ids"]) == request_ids
        and cohort_receipt["active_batch"] == len(merged)
        and cohort_receipt["attribution_kind"] == "isolated"
        and decode_cohort_state["estimator_ingested"]
            == [cohort_id]
        and sum(
            row["measured_energy"] is not None
            for row in request_results
        ) == 1,
        "isolated cohort energy ownership differs",
    )


def _placement_summary(
    request_results: list[dict[str, Any]],
) -> dict[str, object]:
    route_distribution: dict[str, int] = {}
    initial_fraction_distribution: dict[str, int] = {}
    explored_fraction_distribution: dict[str, int] = {}
    selected_fraction_distribution: dict[str, int] = {}
    executed_fraction_distribution: dict[str, int] = {}
    transition_count = 0
    model_reload_count = 0
    for row in request_results:
        terminal = row["terminal_ticket"]
        plan = terminal.get("execution_plan") or {}
        family = str(plan.get("route_family", "unknown"))
        route_key = row["model_id"] + ":" + family
        route_distribution[route_key] = (
            route_distribution.get(route_key, 0) + 1
        )
        fraction_history = row["fraction_history"]
        fraction_sets = (
            (
                initial_fraction_distribution,
                (fraction_history["initial_split_fraction_ppm"],),
            ),
            (
                explored_fraction_distribution,
                fraction_history["explored_split_fractions_ppm"],
            ),
            (
                selected_fraction_distribution,
                (fraction_history["selected_split_fraction_ppm"],),
            ),
            (
                executed_fraction_distribution,
                fraction_history[
                    "physically_executed_split_fractions_ppm"
                ],
            ),
        )
        for distribution, fractions in fraction_sets:
            for fraction in fractions:
                fraction_key = (
                    row["model_id"] + ":" + str(fraction)
                )
                distribution[fraction_key] = (
                    distribution.get(fraction_key, 0) + 1
                )
        receipts = terminal.get("transition_receipts") or []
        transition_count += len(receipts)
        model_reload_count += sum(
            receipt.get("target_state") in {"hot", "warm"}
            for receipt in receipts
        )
    return {
        "adaptive_fraction_counts": dict(sorted(
            selected_fraction_distribution.items()
        )),
        "executed_fraction_counts": dict(sorted(
            executed_fraction_distribution.items()
        )),
        "explored_fraction_counts": dict(sorted(
            explored_fraction_distribution.items()
        )),
        "initial_fraction_counts": dict(sorted(
            initial_fraction_distribution.items()
        )),
        "model_reload_count": model_reload_count,
        "route_counts": dict(sorted(route_distribution.items())),
        "transition_count": transition_count,
    }


@dataclass(frozen=True)
class _TraceRun:
    epoch_ns: int
    paid_end_ns: int
    trace_energy: Any
    journal: _JournalSummary
    adaptive_observations: dict[str, Any]
    request_results: list[dict[str, Any]]
    decode_cohort_state: dict[str, Any]
    observations: dict[str, Any]
    replay_schedule: dict[str, Any]
    active_slot_samples: list[dict[str, object]]
    phone_residency_at_completion: dict[str, object] = field(default_factory=dict)
    phone_residency_phase_events: tuple[dict[str, object], ...] = ()
    phone_residency_call_events: tuple[dict[str, object], ...] = ()
    startup_preloads: tuple[dict[str, object], ...] = ()
    rejections: dict[str, dict[str, Any]] = field(default_factory=dict)
    request_shape_preflight: dict[str, Any] = field(default_factory=dict)


def _capture_phone_residency_evidence(rig):
    state = dict(rig.direct_phone_residency_state)
    return {
        "phone_residency_at_completion": state,
        "phone_residency_phase_events": tuple(rig.phone_residency_phase_events) if state["active"] else (),
        "phone_residency_call_events": tuple(rig.phone_residency_call_events) if state["active"] else (),
    }


def _elastic_drop_result(
    rig: HeterogeneousPhysicalRig,
    request_results: list[dict[str, Any]],
    scheduler: UnifiedScheduler | None = None,
) -> dict[str, Any]:
    """SERVER_EXITED / REQUEST_RECOVERED rows, the device membership events and the helper
    mask events (elastic phones: drop recovery, quarantine, join, S2a mask-out), and under
    elastic phones the thermal gate onsets (THERMAL_DEFERRAL rows).

    Only a run that reaped a server, recovered a request or changed its phone
    membership carries the keys, so every other RESULT stays byte-identical.
    """
    membership_events = getattr(scheduler, "device_membership_events", None)
    membership = (
        [dict(row) for row in membership_events()] if callable(membership_events) else []
    )
    helper_membership = [dict(row) for row in getattr(rig, "helper_membership_events", ())]
    # S2a (helper_loss_recovery: mask_out): helpers masked out of / re-attached to live servers
    helper_masks = [dict(row) for row in getattr(rig, "helper_mask_events", ())]
    exited = [dict(row) for row in getattr(rig, "server_exit_events", ())]
    recovered = [
        dict(row)
        for result in request_results
        for row in result.get("request_recovered", ())
    ]
    thermal_events = getattr(scheduler, "thermal_deferral_events", None)
    thermal = (
        [dict(row) for row in thermal_events()]
        if callable(thermal_events)
        and getattr(getattr(rig, "configuration", None), "elastic_phones", None) is not None
        else []
    )
    require(
        all(row.get("kind") == "SERVER_EXITED" for row in exited)
        and all(row.get("kind") == "REQUEST_RECOVERED" for row in recovered),
        "elastic drop recovery event kinds",
    )
    return {
        **({"server_exited_events": exited} if exited else {}),
        **({"request_recovered_events": recovered} if recovered else {}),
        **({"device_membership_events": membership} if membership else {}),
        **({"helper_membership_events": helper_membership} if helper_membership else {}),
        **({"helper_mask_events": helper_masks} if helper_masks else {}),
        **({"thermal_deferral_events": thermal} if thermal else {}),
    }


def _note_next_arrival(rig: object, arrival_us: int | None) -> None:
    """Feed the arrival calendar to the device power controller.

    Only a rig configured with a ``device_power`` policy is told; a run without the policy
    (and test rigs without the method) is byte-identical to before.
    """
    if getattr(getattr(rig, "configuration", None), "device_power", None) is None:
        return
    rig.note_next_arrival(arrival_us)


def _device_power_result(rig: HeterogeneousPhysicalRig) -> dict[str, Any]:
    """DEVICE_POWER_STATE rows of the device power controller. Only a run with a ``device_power``
    policy carries the key (also when the controller went UNAVAILABLE), so every other RESULT
    stays byte-identical."""
    if getattr(getattr(rig, "configuration", None), "device_power", None) is None:
        return {}
    events = [dict(row) for row in getattr(rig, "device_power_events", ())]
    require(all(row.get("kind") == "DEVICE_POWER_STATE" for row in events), "device power event kinds")
    return {"device_power_events": events}


def _speculative_rows_result(
    rig: HeterogeneousPhysicalRig, request_results: list[dict[str, Any]]
) -> dict[str, Any]:
    """The ``speculative_rows`` summary: the campaign configuration and the draft statistics
    summed over the requests that carried them. Only a run configured with the key carries it,
    so every other RESULT stays byte-identical."""
    configuration = getattr(getattr(rig, "configuration", None), "speculative_rows", None)
    if configuration is None:
        return {}
    rows = [result["speculative"] for result in request_results if type(result.get("speculative")) is dict]
    drafted = [row for row in rows if row.get("draft_n", 0) > 0]
    output_tokens = sum(result["output_tokens"] for result in request_results
                        if type(result.get("speculative")) is dict)
    steps = sum(row["verif_steps"] for row in rows)
    draft_n = sum(row["draft_n"] for row in rows)
    accepted = sum(row["draft_accepted"] for row in rows)
    return {"speculative_rows": {
        "acceptance_rate_ppm": 0 if draft_n == 0 else accepted * 1_000_000 // draft_n,
        "configuration": {model_id: row.to_json() for model_id, row in sorted(configuration.items())},
        "draft_accepted": accepted,
        "draft_n": draft_n,
        "requests_with_draft": len(drafted),
        "requests_with_statistics": len(rows),
        "schema": "s42-speculative-rows-summary-v1",
        "tokens_per_step_ppm": 0 if steps == 0 else output_tokens * 1_000_000 // steps,
        "verif_steps": steps,
    }}


def _assemble_result(
    args: argparse.Namespace,
    models: _TraceModels,
    manifests: dict[str, ModelManifest],
    host_dependencies: dict[str, Path],
    scheduler: UnifiedScheduler,
    rig: HeterogeneousPhysicalRig,
    coordinator: CanonicalArrivalCoordinator,
    merged: list[dict[str, Any]],
    run: _TraceRun,
) -> dict[str, Any]:
    catalog = models.catalog
    expected_qwen = models.expected_qwen
    expected_gemma = models.expected_gemma
    llama_model_id = models.llama_model_id
    records = run.journal.records
    request_results = run.request_results
    placement_summary = _placement_summary(request_results)
    return {
        **_elastic_drop_result(rig, request_results, scheduler),
        **_device_power_result(rig),
        **_speculative_rows_result(rig, request_results),
        "candidate_device_families": [
            list(row) for row in sorted(candidate_device_families(records))
        ],
        "catalog_id": catalog.catalog_id,
        "catalog_sha256": digest(args.capability_catalog),
        "counts": {
            "decisions": len(run.journal.decisions),
            "execution_attempts": len(run.journal.attempts),
            "gemma": sum(
                item["model_id"] == expected_gemma.model_id
                for item in merged
            ),
            "llama": sum(
                item["model_id"] == llama_model_id for item in merged
            ),
            "qwen": sum(
                item["model_id"] == expected_qwen.model_id
                for item in merged
            ),
            "rejected": len(run.rejections),
            "requests": len(request_results),
            "submitted": len(merged),
            "terminals": len(run.journal.terminals),
        },
        "decode_cohort_state": run.decode_cohort_state,
        "duration_us": (run.paid_end_ns - run.epoch_ns) // 1000,
        "energy_accounting": trace_energy_accounting(run.trace_energy),
        "journal_final_hash": run.journal.journal["head_record_sha256"],
        "journal_path": "SCHEDULER_DECISION_LOG.json",
        "model_artifacts": {
            model_id: {
                "artifact_bytes": manifest.artifact_bytes,
                "artifact_sha256": manifest.artifact_sha256,
            }
            for model_id, manifest in sorted(manifests.items())
        },
        "model_roles": {
            "gemma": expected_gemma.model_id,
            "llama": llama_model_id,
            "qwen": expected_qwen.model_id,
        },
        "maximum_latency_ppm": catalog.maximum_latency_ppm,
        "maximum_phone_sessions": args.maximum_phone_sessions,
        "model_placement_controller_stats": dict(
            scheduler.model_placement_controller_stats()
        ),
        "model_placement_events": [
            dict(row) for row in scheduler.model_placement_events()
        ],
        "helper_preparation_fault_injection": (
            coordinator.helper_preparation_fault_injector.to_json()
        ),
        "phone_residency_events": [
            dict(row) for row in scheduler.phone_residency_events()
        ],
        "request_helper_events": [
            dict(row) for row in scheduler.request_helper_events()
        ],
        "adaptive_timing_events": list(rig.adaptive_timing_events),
        "background_placement_stats": dict(
            scheduler.background_placement_stats()
        ),
        "model_placement_epoch_stats": dict(
            scheduler.model_placement_epoch_stats()
        ),
        "placement_summary": placement_summary,
        "observation_store_path": "AUTOMATED_OBSERVATIONS.json",
        "observation_store_sha256": run.observations["store_sha256"],
        "observation_store_state": dict(
            scheduler.automated_observation_state()
        ),
        "adaptive_observation_store_path": (
            "ADAPTIVE_DECODE_OBSERVATIONS.json"
        ),
        "adaptive_observation_store_sha256": (
            run.adaptive_observations["store_sha256"]
        ),
        "adaptive_observation_store_state": dict(
            scheduler.adaptive_decode_observation_state()
        ),
        "adaptive_decode_overrides": _adaptive_decode_overrides_from_json(
            getattr(args, "adaptive_decode_overrides_json", None)
        ) or None,
        **({} if getattr(args, "dispatch_policy_json", None) is None else {
            "dispatch_policy": dict(scheduler.runtime_dispatch_policy_state()),
        }),
        "adaptive_maximum_probe_attempts_per_context": (
            args.adaptive_maximum_probe_attempts_per_context
        ),
        "adaptive_minimum_remaining_tokens": (
            args.adaptive_minimum_remaining_tokens
        ),
        "active_server_slots": sorted(
            run.active_slot_samples,
            key=lambda row: (
                row["observed_at_us"], row["request_id"]
            ),
        ),
        "active_slots_peak": max(
            (
                int(row["active_slots"])
                for row in run.active_slot_samples
            ),
            default=0,
        ),
        "paid_end_ns": run.paid_end_ns,
        "paid_start_ns": run.epoch_ns,
        "rejected_requests": [
            run.rejections[request_id]
            for request_id in sorted(
                run.rejections,
                key=lambda key: (
                    run.rejections[key]["arrival_us"], key
                ),
            )
        ],
        "request_results": request_results,
        "request_shape_preflight": run.request_shape_preflight,
        "replay_schedule": run.replay_schedule,
        "request_candidate_coverage": [
            row.to_json() for row in run.journal.candidate_coverage
        ],
        "scheduler_decision_timings": [
            dict(row) for row in scheduler.runtime_decision_timings()
        ],
        "schema": RESULT_SCHEMA,
        "selection_mode": args.selection_mode,
        "fixed_phone_residency": scheduler.fixed_phone_residency_configuration(),
        "offline_phone_residency": (
            None if scheduler.offline_phone_residency_snapshot() is None else
            dict(scheduler.offline_phone_residency_snapshot())
        ),
        "phone_residency_at_completion": run.phone_residency_at_completion,
        "phone_residency_phase_events": list(run.phone_residency_phase_events),
        "phone_residency_call_events": list(run.phone_residency_call_events),
        "preparation_accounting": (
            "first-host-sample-before-runtime-load-through-dynamic-endpoint-cleanup"
            if getattr(args, "include_startup_preparation", False) else
            "trace-start-after-common-runtime-warmup-through-terminal-cleanup"
        ),
        "startup_desktop_preloads": list(run.startup_preloads),
        "initial_observation_inputs": {
            name: None if getattr(args, name, None) is None else
            "sha256:" + digest(getattr(args, name))
            for name in (
                "observation_store_input", "observation_source_catalog",
                "adaptive_observation_store_input", "adaptive_observation_source_catalog",
            )
        },
        "adaptive_controller_configuration": (
            _configured_adaptive_decode_config(args) or AdaptiveDecodeConfig()
        ).to_json(),
        "execution_identity": {
            "cuda_graph_mode_by_artifact": {
                row.artifact_sha256: row.cuda_graph_mode
                for row in models.catalog.desktop_control_profiles
            },
            "desktop_launch_mode_by_artifact": {
                row.artifact_sha256: models.catalog.composite_executor_by_id[
                    row.executor_id
                ].adapter_parameters.get("desktop_launch_mode", "canonical")
                for row in models.catalog.desktop_control_profiles
            },
            "binaries": {
                "bridge": "sha256:" + digest(args.bridge),
                "close_helper": "sha256:" + digest(
                    args.close_helper
                ),
                "resident_server": "sha256:" + digest(
                    args.resident_server
                ),
                "server": "sha256:" + digest(args.server),
                **{
                    "host_dependency:" + name: "sha256:" + digest(path)
                    for name, path in sorted(host_dependencies.items())
                },
            },
            "source_manifest_sha256": (
                None
                if args.source_manifest is None
                else "sha256:" + digest(args.source_manifest)
            ),
        },
        # PARTIAL: every submitted request reached a terminal state but some
        # were rejected at submission as permanently unsupported; the
        # original workload was not completed and no completion claim holds.
        "status": "PASS" if not run.rejections else "PARTIAL",
        "trace_energy": measured_energy(run.trace_energy),
        "trace_identity": {
            "large_sha256": digest(args.large_requests),
            "overlay_sha256": digest(args.overlay_requests),
        },
        "transport_receipts": list(rig.bridge_terminal_receipts),
        "direct_phone_receipts": list(rig.direct_phone_receipts),
        # decode-only relocation admission (rig-level ledger, release credits, prompt holds); None when no budget
        "dormant_share_admission": (
            None if getattr(rig, "dormant_share_coordinator", lambda: None)() is None
            else rig.dormant_share_coordinator().to_json()
        ),
        "android_control_events": list(rig.android_control_events),
        "physical_execution_proofs": dict(rig.execution_proofs),
        "transport_qualifications": list(
            rig.transport_qualifications
        ),
        "usb_restore_receipts": list(rig.usb_restore_receipts),
    }


def _write_run_artifacts(
    output: Path,
    journal: dict[str, Any],
    observations: dict[str, Any],
    adaptive_observations: dict[str, Any],
    result: dict[str, Any],
) -> None:
    (output / "SCHEDULER_DECISION_LOG.json").write_bytes(
        canonical(journal)
    )
    (output / "AUTOMATED_OBSERVATIONS.json").write_bytes(
        canonical(observations)
    )
    (output / "ADAPTIVE_DECODE_OBSERVATIONS.json").write_bytes(
        canonical(adaptive_observations)
    )
    (output / "RESULT.json").write_bytes(canonical(result))


def _write_failure_artifacts(
    output: Path, scheduler: UnifiedScheduler, error: BaseException
) -> None:
    if not output.exists():
        return
    try:
        (output / "FAILURE_SCHEDULER_DECISION_LOG.json").write_bytes(
            canonical(scheduler.runtime_decision_log())
        )
    except BaseException:
        pass
    try:
        (output / "FAILURE_AUTOMATED_OBSERVATIONS.json").write_bytes(
            canonical(dict(scheduler.automated_observation_snapshot()))
        )
    except BaseException:
        pass
    try:
        (output / "FAILURE_ADAPTIVE_DECODE_OBSERVATIONS.json").write_bytes(
            canonical(dict(
                scheduler.adaptive_decode_observation_snapshot()
            ))
        )
    except BaseException:
        pass
    try:
        (output / "FAILURE_REQUEST_HELPER_EVENTS.json").write_bytes(canonical([
            dict(row) for row in scheduler.request_helper_events()
        ]))
    except BaseException:
        pass
    (output / "FAILURE.json").write_bytes(canonical({
        "error": f"{type(error).__name__}: {error}",
        "schema": "s42-unified-fp16-llama-overlay-failure-v1",
        "status": "FAIL",
        "traceback": traceback.format_exc(),
    }))


def _close_rig(
    output: Path,
    rig: HeterogeneousPhysicalRig,
    coordinator: CanonicalArrivalCoordinator | None,
    primary_error: BaseException | None,
) -> None:
    cleanup_errors = []
    try:
        if coordinator is not None:
            coordinator.close(wait=primary_error is None)
    except BaseException as error:
        cleanup_errors.append(error)
    try:
        timing_path = output / "adaptive-timing-events.json"
        if not timing_path.exists():
            with timing_path.open("xb") as stream:
                stream.write(canonical(list(rig.adaptive_timing_events)))
    except BaseException as error:
        cleanup_errors.append(error)
    try:
        rig.close(require_phone_execution=primary_error is None)
    except BaseException as error:
        cleanup_errors.append(error)
    try:
        sample_path = output / "resource-samples.jsonl"
        if not sample_path.exists():
            with sample_path.open("xb") as stream:
                for row in rig.host_samples:
                    stream.write(canonical(row))
    except BaseException as error:
        cleanup_errors.append(error)
    try:
        diagnostics_path = output / "phone-power-diagnostics.json"
        if not diagnostics_path.exists():
            diagnostics_path.write_bytes(canonical(
                rig.phone_power_diagnostics
            ))
    except BaseException as error:
        cleanup_errors.append(error)
    try:
        diagnostics_path = output / "host-power-diagnostics.json"
        if not diagnostics_path.exists():
            diagnostics_path.write_bytes(canonical(
                rig.host_power_diagnostics
            ))
    except BaseException as error:
        cleanup_errors.append(error)
    if cleanup_errors:
        message = "; ".join(
            f"{type(error).__name__}: {error}"
            for error in cleanup_errors
        )
        if primary_error is None:
            raise UnifiedTraceError(
                "physical cleanup failed: " + message
            ) from cleanup_errors[0]
        primary_error.add_note("physical cleanup failed: " + message)
        try:
            (output / "CLEANUP_FAILURE.json").write_bytes(canonical({
                "error": message,
                "primary_error": (
                    f"{type(primary_error).__name__}: {primary_error}"
                ),
                "schema": "s42-unified-cleanup-failure-v1",
                "status": "FAIL",
            }))
        except BaseException:
            pass


def main() -> int:
    args = _build_parser().parse_args()

    host_dependencies = _validate_arguments(args)
    startup_parents = tuple(StartupDesktopParentConfiguration.from_json(value)
                            for value in json.loads(args.startup_desktop_parents_json or "[]"))
    require(not startup_parents or args.include_startup_preparation,
            "startup desktop preload requires paid preparation accounting")
    require(len({row.executor_id for row in startup_parents}) == len(startup_parents),
            "startup desktop executors are duplicated")

    models = _load_trace_models(args)
    scheduler, manifests, initial_adaptive_group_sha256s = _build_scheduler(
        args, models
    )
    aliases, merged, replay_schedule, isolated_cohort_expected = (
        _select_replay(args, models, manifests)
    )

    args.output.mkdir(parents=True)
    persist_replay_schedule(
        args.output / "REPLAY_SCHEDULE.json", replay_schedule
    )
    streams = args.output / "streams"
    streams.mkdir()
    snapshots = args.output / "snapshots"
    snapshots.mkdir()
    rig = _build_rig(args, models, manifests, host_dependencies)
    direct_phone_preflight = rig.direct_phone_preflight()
    (args.output / "DIRECT_PHONE_PREFLIGHT.json").write_bytes(canonical(
        direct_phone_preflight.to_json()
    ))
    warm_payload = _warm_payload(models, aliases, streams)
    coordinator = None
    state = _ArrivalState()
    primary_error: BaseException | None = None
    preload_pool = None
    preload_future = None
    persist_runtime_snapshot = _snapshot_provider(rig, scheduler, snapshots)

    try:
        # Every trace request is screened for a permanently unsupported
        # shape before the paid interval starts; verdicts are published in
        # RESULT.request_shape_preflight. Unsupported requests are still
        # submitted at their arrival so the rejection is recorded with the
        # scheduler's own exact reason.
        request_shape_preflight = _screen_request_shapes(scheduler, merged)
        for row in request_shape_preflight["unsupported"]:
            print(
                "UNSUPPORTED SHAPE " + row["request_id"] + " ("
                + str(row["reason"]) + ")",
                flush=True,
            )
        rig.start(warm_payload)
        epoch_ns = (
            int(rig.host_samples[0]["t_ns"])
            if getattr(args, "include_startup_preparation", False)
            else time.monotonic_ns()
        )
        rig.begin_trace(epoch_ns)
        startup_preloads = _preload_startup_parents(
            args, scheduler, rig, aliases, streams, epoch_ns, startup_parents,
        )
        if scheduler.fixed_phone_residency_configuration() is not None:
            preload_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fixed-residency")
            preload_future = preload_pool.submit(
                _preload_fixed_residency, args, scheduler, rig, aliases, streams, epoch_ns,
            )
        coordinator = CanonicalArrivalCoordinator(
            scheduler,
            None,
            epoch_ns=epoch_ns,
            snapshot_provider=persist_runtime_snapshot,
            max_workers=args.max_workers,
            backend_factory=lambda _ticket, _payload: rig.backend(),
            helper_preparation_fault_injection=(
                args.helper_preparation_fault_injection
            ),
            fail_fast=args.lifecycle_failure_mode == "fail-fast",
        )
        _submit_arrivals(
            args,
            merged,
            coordinator,
            rig,
            epoch_ns,
            streams,
            snapshots,
            aliases,
            state,
        )
        completed = coordinator.drain(timeout_s=14_400)
        if preload_future is not None:
            preload_future.result(timeout=300)
        phone_residency_evidence = _capture_phone_residency_evidence(rig)
        rig.end_trace(require_phone_execution=(
            scheduler.fixed_phone_residency_configuration() is None
        ))
        paid_end_ns = time.monotonic_ns()
        trace_energy = rig.trace_energy(epoch_ns, paid_end_ns)
        journal = _summarize_journal(
            args, scheduler, models.catalog, manifests, merged,
            startup_request_ids=tuple(row["request_id"] for row in startup_preloads),
            rejected_request_ids=frozenset(state.rejections),
        )
        adaptive_observations = dict(
            scheduler.adaptive_decode_observation_snapshot()
        )
        current_adaptive_groups = [
            group
            for group in adaptive_observations.get("groups", [])
            if group["grouped_observation_sha256"]
                not in initial_adaptive_group_sha256s
        ]
        request_results = _build_request_results(
            scheduler, completed, merged, current_adaptive_groups, state
        )
        decode_cohort_state = dict(
            scheduler.runtime_decode_cohort_snapshot()
        )
        if isolated_cohort_expected:
            _require_isolated_cohort(
                decode_cohort_state, merged, request_results
            )
        observations = dict(scheduler.automated_observation_snapshot())
        result = _assemble_result(
            args,
            models,
            manifests,
            host_dependencies,
            scheduler,
            rig,
            coordinator,
            merged,
            _TraceRun(
                epoch_ns=epoch_ns,
                paid_end_ns=paid_end_ns,
                trace_energy=trace_energy,
                journal=journal,
                adaptive_observations=adaptive_observations,
                request_results=request_results,
                decode_cohort_state=decode_cohort_state,
                observations=observations,
                replay_schedule=replay_schedule,
                active_slot_samples=state.active_slot_samples,
                startup_preloads=tuple(startup_preloads),
                rejections=dict(state.rejections),
                request_shape_preflight=request_shape_preflight,
                **phone_residency_evidence,
            ),
        )
        _write_run_artifacts(
            args.output,
            journal.journal,
            observations,
            adaptive_observations,
            result,
        )
        print(json.dumps({
            "counts": result["counts"],
            "duration_us": result["duration_us"],
            "selection_mode": args.selection_mode,
            "status": result["status"],
        }, sort_keys=True))
        return 0
    except BaseException as error:
        primary_error = error
        _write_failure_artifacts(args.output, scheduler, error)
        raise
    finally:
        if preload_pool is not None:
            preload_pool.shutdown(wait=True, cancel_futures=True)
        _close_rig(args.output, rig, coordinator, primary_error)


if __name__ == "__main__":
    raise SystemExit(main())
