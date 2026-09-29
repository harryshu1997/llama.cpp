import json
from pathlib import Path
import traceback

from research_dev.scheduler import (
    HeterogeneousRuntimeSnapshot,
    Request,
    RuntimeCapabilityCatalog,
    UnifiedScheduler,
)
from research_dev.scheduler.campaigns.burstgpt.runner import (
    GEMMA_ROLE,
    QWEN_ROLE,
    apply_named_replay_schedule,
    load_object,
    merge_rows,
    validate_trace,
)


root = Path("/home/zhihao")
repo = root / "llama.cpp-release"
run_root = root / "s42-authoritative-phone-residency-adaptive-20260830-v2"
evidence_root = (
    root / "s42-route-evidence-five-prefix-20260829-v6-retained-inactive-shards"
)
catalog = RuntimeCapabilityCatalog.from_json(
    load_object(run_root / "inputs/UNIFIED_RUNTIME_CATALOG.json")
)
source_catalog = RuntimeCapabilityCatalog.from_json(
    load_object(evidence_root / "UNIFIED_RUNTIME_CATALOG.json")
)
scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
scheduler.register_runtime_capabilities(catalog)
cache = root / ".cache/llama.cpp/research-scheduler/gguf-manifests.json"
for model_id, path in {
    "qwen3-14b-q4km-dequant-f16": (
        root / "models/Qwen3-14B-Q4KM-dequant-f16.gguf"
    ),
    "gemma-4-12b-q40-dequant-f16": (
        root / "models/gemma-4-12B-Q40-dequant-f16.gguf"
    ),
}.items():
    scheduler.register_gguf_model(model_id, path, cache_path=cache)
scheduler.load_automated_observations(
    load_object(evidence_root / "run/AUTOMATED_OBSERVATIONS.json"),
    source_catalog=source_catalog,
)
scheduler.load_adaptive_decode_observations(
    load_object(evidence_root / "run/ADAPTIVE_DECODE_OBSERVATIONS.json"),
    source_catalog=source_catalog,
)
compiler = scheduler._automated_compiler()
original_subset_evidence = compiler.phone_residency_subset_evidence


def diagnostic_subset_evidence(manifest, candidate):
    result = original_subset_evidence(manifest, candidate)
    shards = tuple(
        row for row in candidate.plan.execution_contract.phone_shards
        if row.artifact_sha256 in {None, manifest.artifact_sha256}
    )
    target_mask = 0
    for shard in shards:
        target_mask |= shard.layer_mask
    evidence = compiler._phone_residency_route_evidence.get(
        manifest.artifact_sha256
    )
    if result is None and len(shards) == 3 and target_mask == 131071:
        print(json.dumps({
            "subset_debug": {
                "evidence": None if evidence is None else {
                    "component": evidence.source_component_capability_sha256,
                    "desktop": evidence.source_desktop_placement_sha256,
                    "endpoint": evidence.source_endpoint,
                    "executor": evidence.source_executor_id,
                    "layer_mask": evidence.source_layer_mask,
                    "maximum_columns": evidence.source_maximum_columns,
                    "protocol": evidence.source_operator_plan_protocol,
                    "sessions": list(evidence.source_session_ids),
                    "batch_plan": evidence.source_batch_plan,
                    "maximum_batch_size": evidence.source_maximum_batch_size,
                    "queue_depth": evidence.source_queue_depth,
                },
                "target": {
                    "component": compiler.component_capability_identity(
                        candidate.plan, candidate.binding.executor_id
                    ),
                    "desktop": candidate.plan.desktop_placement_sha256,
                    "endpoint": candidate.binding.endpoint,
                    "executor": candidate.binding.executor_id,
                    "layer_mask": target_mask,
                    "maximum_columns": max(
                        row.maximum_columns for row in shards
                    ),
                    "protocol": candidate.binding.operator_plan_protocol,
                    "sessions": sorted(row.session_id for row in shards),
                    "batch_plan": candidate.plan.execution_contract.batch_plan,
                    "maximum_batch_size": (
                        candidate.plan.execution_contract.maximum_batch_size
                    ),
                    "queue_depth": candidate.plan.execution_contract.queue_depth,
                },
            },
        }, sort_keys=True))
    return result


compiler.phone_residency_subset_evidence = diagnostic_subset_evidence
original_history = (
    scheduler._adaptive_decode
    .historical_route_estimate_with_assumed_phone_power
)


def diagnostic_history(*args, **kwargs):
    result = original_history(*args, **kwargs)
    if kwargs.get("operator_subset_prior", False):
        source_mask = kwargs["operator_subset_source_layer_mask"]
        artifact = kwargs["model_artifact_sha256"]
        desktop = kwargs["baseline"].desktop_placement_sha256
        active_batch = kwargs["active_batch"]
        matching_groups = [
            group for group in scheduler._adaptive_decode._history.values()
            if group.terminal_status == "COMPLETED"
            and group.model_artifact_sha256 == artifact
            and group.desktop_placement_sha256 == desktop
            and any(
                not window.policy.baseline
                and window.policy.layer_mask == source_mask
                for window in group.windows
            )
        ]
        counts = {}
        for group in matching_groups:
            for window in group.windows:
                if window.active_batch != active_batch:
                    continue
                key = (
                    window.context_length.bit_length(),
                    window.policy.split_fraction_ppm,
                    window.policy.baseline,
                    window.energy_measurement_eligible,
                )
                counts[key] = counts.get(key, 0) + 1
        print(json.dumps({
            "subset_history": {
                "counts": [
                    {"active": key[3], "baseline": key[2],
                     "bucket": key[0], "count": value,
                     "fraction": key[1]}
                    for key, value in sorted(counts.items())
                ],
                "group_count": len(matching_groups),
                "policies": [
                    {
                        "columns": row.columns,
                        "fraction": row.split_fraction_ppm,
                        "layer_mask": row.layer_mask,
                    }
                    for row in kwargs["candidates"]
                ],
                "result": None if result is None else {
                    "evidence_match": result.evidence_match,
                    "energy_lower": result.baseline_energy_lower_per_token_uj,
                    "energy_upper": result.selected_energy_upper_per_token_uj,
                    "source_layer_mask": result.source_layer_mask,
                    "target_layer_mask": result.target_layer_mask,
                },
                "source_layer_mask": source_mask,
            },
        }, sort_keys=True))
    return result


scheduler._adaptive_decode.historical_route_estimate_with_assumed_phone_power = (
    diagnostic_history
)
original_apply_history = scheduler._apply_adaptive_history_costs


def diagnostic_apply_history(candidate_set, request, manifest, cost_features=None):
    before = []
    for candidate in candidate_set.candidates:
        shards = candidate.plan.execution_contract.phone_shards
        mask = 0
        for shard in shards:
            if shard.artifact_sha256 in {None, manifest.artifact_sha256}:
                mask |= shard.layer_mask
        if len(shards) == 3 and mask == 131071:
            coordinator = (
                scheduler._runtime_capabilities.composite_executor_by_id.get(
                    candidate.binding.executor_id
                )
            )
            before.append({
                "energy_evidence": candidate.cost.energy_evidence,
                "endpoint": candidate.binding.endpoint,
                "executor": candidate.binding.executor_id,
                "maturity": candidate.maturity,
                "rejections": list(candidate.rejection_reasons),
                "coordinator_maturity": (
                    None if coordinator is None else coordinator.maturity
                ),
                "coordinator_evidence": (
                    [] if coordinator is None else list(coordinator.evidence_ids)
                ),
                "device_maturities": {
                    device_id: scheduler._runtime_capabilities
                        .executor_by_device[device_id].maturity
                    for device_id in candidate.device_ids
                },
                "transition_maturities": [
                    row.maturity for row in candidate.plan.transitions
                ],
            })
    result = original_apply_history(
        candidate_set, request, manifest, cost_features
    )
    if before:
        after = []
        for candidate in result.candidates:
            shards = candidate.plan.execution_contract.phone_shards
            mask = 0
            for shard in shards:
                if shard.artifact_sha256 in {None, manifest.artifact_sha256}:
                    mask |= shard.layer_mask
            if len(shards) == 3 and mask == 131071:
                after.append({
                    "evidence_match": dict(
                        candidate.residency_break_even or {}
                    ).get("adaptive_history_evidence_match"),
                    "maturity": candidate.maturity,
                    "rejections": list(candidate.rejection_reasons),
                })
        print(json.dumps({
            "apply_history": {"after": after, "before": before},
        }, sort_keys=True))
    return result


scheduler._apply_adaptive_history_costs = diagnostic_apply_history
large, overlay = validate_trace(
    repo / (
        "research_dev/spikes/s41_gemma_qwen_continuous_baseline/"
        "tp_operator_split_v1/burstgpt_gpu_cpu_op15_v1/"
        "REQUESTS_SEMANTIC_SOURCE.jsonl"
    ),
    repo / (
        "research_dev/spikes/s42_general_energy_scheduler_v1/"
        "full_fp16_burstgpt_v1/small_model_overlay_v1/"
        "REQUESTS_LLAMA1B_10.jsonl"
    ),
    load_object(
        repo / (
            "research_dev/spikes/s42_general_energy_scheduler_v1/"
            "full_fp16_burstgpt_v1/small_model_overlay_v1/"
            "TRACE_MANIFEST.json"
        )
    ),
)
merged = merge_rows(
    large,
    overlay,
    {
        QWEN_ROLE: "qwen3-14b-q4km-dequant-f16",
        GEMMA_ROLE: "gemma-4-12b-q40-dequant-f16",
    },
)
selected, _schedule = apply_named_replay_schedule(
    merged,
    load_object(run_root / "inputs/REPLAY_SCHEDULE.json"),
)

results = []
qwen_indices = {34, 35, 38, 39, 40, 41, 42, 43}
for item in (
    row for row in selected if row["combined_index"] in qwen_indices
):
    index = item["combined_index"]
    source = item["row"]
    pre_evidence_status = dict(
        compiler.phone_residency_evidence_status(
            scheduler.runtime_model_manifest(item["model_id"])
                .artifact_sha256
        )
    )
    snapshot = HeterogeneousRuntimeSnapshot.from_json(
        load_object(run_root / f"run/snapshots/request-{index:03d}.json")
    )
    request = Request(
        request_id=source["event_id"],
        workload_id="physical:" + item["model_id"],
        arrival_us=source["arrival_us"],
        deadline_us=source["arrival_us"] + source["slo_us"],
        input_tokens=source["input_tokens"],
        output_tokens=source["output_tokens"],
        quality_requirement="semantic",
    )
    try:
        ticket = scheduler.submit_automated_request(
            request,
            item["model_id"],
            snapshot,
            observed_at_us=snapshot.captured_at_us,
            selection_mode="energy-aware",
        )
    except Exception as error:
        print(json.dumps({
            "error": type(error).__name__ + ":" + str(error),
            "index": index,
            "partial": results,
            "traceback": traceback.format_exc(),
        }, sort_keys=True))
        raise SystemExit(1)
    record = next(
        row
        for row in reversed(scheduler.runtime_decision_log()["records"])
        if request.request_id in row["request_ids"]
        and row["event_kind"] in {"DECISION", "REPLAN"}
    )
    selected_row = next(
        row for row in record["candidates"]
        if row["route_id"] == ticket.decision.route_id
    )
    details = selected_row.get("details") or {}
    break_even = details.get("residency_break_even") or {}
    phone_rows = []
    for row in record["candidates"]:
        candidate_details = row.get("details") or {}
        candidate_break_even = (
            candidate_details.get("residency_break_even") or {}
        )
        shards = (
            ((candidate_details.get("operator_plan") or {})
             .get("execution_contract") or {})
            .get("phone_shards") or []
        )
        if len(shards) != 3:
            continue
        phone_rows.append({
            "admitted": row["admitted"],
            "evidence_match": candidate_break_even.get(
                "adaptive_history_evidence_match"
            ),
            "geometry": (
                (candidate_details.get("adapter_parameters") or {})
                .get("phone_shard_set_geometry_sha256")
            ),
            "maturity": candidate_details.get("maturity"),
            "rejections": candidate_details.get("rejection_reasons"),
            "route_id": row["route_id"],
            "source_layer_mask": candidate_break_even.get(
                "adaptive_history_source_layer_mask"
            ),
            "target_layer_mask": candidate_break_even.get(
                "adaptive_history_target_layer_mask"
            ),
        })
    state = scheduler._model_placement_controller.planning_phone_layout()
    diagnostic_candidates = []
    if False and index in {42, 43}:
        candidate_set = scheduler.generate_automated_candidates(
            request,
            item["model_id"],
            snapshot,
            observed_at_us=snapshot.captured_at_us,
        )
        candidate_set = scheduler._apply_adaptive_history_costs(
            candidate_set,
            request,
            scheduler.runtime_model_manifest(item["model_id"]),
            snapshot.cost_features,
        )
        candidate_set = (
            scheduler._apply_phone_residency_portfolio_authorization(
                candidate_set,
                scheduler.runtime_model_manifest(item["model_id"]),
                request,
            )
        )
        compiler = scheduler._automated_compiler()
        for candidate in candidate_set.candidates:
            shards = candidate.plan.execution_contract.phone_shards
            if len(shards) != 3:
                continue
            evidence = candidate.residency_break_even or {}
            diagnostic_candidates.append({
                "admitted": candidate.admitted,
                "columns": max(row.maximum_columns for row in shards),
                "component": compiler.component_capability_identity(
                    candidate.plan, candidate.binding.executor_id
                ),
                "desktop_placement": (
                    candidate.plan.desktop_placement_sha256
                ),
                "energy_evidence": candidate.cost.energy_evidence,
                "evidence_match": evidence.get(
                    "adaptive_history_evidence_match"
                ),
                "geometry": candidate.plan.adapter_parameters.get(
                    "phone_shard_set_geometry_sha256"
                ),
                "layer_mask": sum(row.layer_mask for row in shards),
                "maturity": candidate.maturity,
                "rejections": candidate.rejection_reasons,
                "route_id": candidate.candidate_id,
                "source_layer_mask": evidence.get(
                    "adaptive_history_source_layer_mask"
                ),
                "target_layer_mask": evidence.get(
                    "adaptive_history_target_layer_mask"
                ),
            })
    results.append({
        "diagnostic_candidates": diagnostic_candidates,
        "index": index,
        "layout": None if state is None else state.to_json(),
        "model_id": item["model_id"],
        "pre_evidence_status": pre_evidence_status,
        "phone_candidates": phone_rows,
        "selected_evidence_match": break_even.get(
            "adaptive_history_evidence_match"
        ),
        "selected_executor": ticket.binding.executor_id,
        "selected_route": ticket.decision.route_id,
        "ticket_layout_generation": ticket.phone_layout_generation,
        "ticket_projection": (
            None
            if ticket.residency_projection_token is None
            else ticket.residency_projection_token.to_json()
        ),
        "transition_count": len(ticket.execution_plan.transitions),
    })

print(json.dumps({
    "final": results[-1],
    "selected_routes": [
        {
            "index": row["index"],
            "layout_generation": row["ticket_layout_generation"],
            "route": row["selected_route"],
        }
        for row in results
    ],
}, sort_keys=True))
