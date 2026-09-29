import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

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
    canonical,
    load_object,
    merge_rows,
    validate_trace,
)


def plain(value):
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


root = Path("/home/zhihao")
repo = root / "llama.cpp-release"
deploy = root / "s42-mixed-residency-priority-deploy-20260828-v1"
gate = root / "s42-route-evidence-catalog-20260829-v5-artifact-bound"
snapshot_gate = root / "s42-route-evidence-five-prefix-20260829-v1"
inputs = root / "s42-sparse-locality24-matched-ab-20260828-v1/inputs"
output = gate / "FIVE_REQUEST_CAUSAL_QUEUE_REPLAY_V12.json"
if output.exists():
    raise SystemExit("refusing to overwrite " + str(output))

catalog_value = load_object(gate / "UNIFIED_RUNTIME_CATALOG.json")
catalog = RuntimeCapabilityCatalog.from_json(catalog_value)
source_catalog = RuntimeCapabilityCatalog.from_json(
    load_object(inputs / "OBSERVATION_SOURCE_CATALOG.json")
)
scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
scheduler.register_runtime_capabilities(catalog)
cache = root / ".cache/llama.cpp/research-scheduler/gguf-manifests.json"
models = {
    "qwen3-14b-q4km-dequant-f16": (
        root / "models/Qwen3-14B-Q4KM-dequant-f16.gguf"
    ),
    "gemma-4-12b-q40-dequant-f16": (
        root / "models/gemma-4-12B-Q40-dequant-f16.gguf"
    ),
}
for model_id, path in models.items():
    scheduler.register_gguf_model(model_id, path, cache_path=cache)
scheduler.load_automated_observations(
    load_object(inputs / "AUTOMATED_OBSERVATIONS.json"),
    source_catalog=source_catalog,
)
scheduler.load_adaptive_decode_observations(
    load_object(inputs / "ADAPTIVE_DECODE_OBSERVATIONS.json"),
    source_catalog=source_catalog,
)

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
selected, replay = apply_named_replay_schedule(
    merged, load_object(deploy / "PREFIX_REPLAY_SCHEDULE.json")
)

requests = []
phone_cursor = 0
placement_cursor = 0
for item in selected:
    index = item["combined_index"]
    source_row = item["row"]
    model_id = item["model_id"]
    snapshot = HeterogeneousRuntimeSnapshot.from_json(
        load_object(snapshot_gate / f"run/snapshots/request-{index:03d}.json")
    )
    request = Request(
        request_id=source_row["event_id"],
        workload_id="physical:" + model_id,
        arrival_us=source_row["arrival_us"],
        deadline_us=source_row["arrival_us"] + source_row["slo_us"],
        input_tokens=source_row["input_tokens"],
        output_tokens=source_row["output_tokens"],
        quality_requirement="semantic",
    )
    ticket = scheduler.submit_automated_request(
        request,
        model_id,
        snapshot,
        observed_at_us=snapshot.captured_at_us,
        selection_mode="energy-aware",
    )
    record = next(
        record
        for record in reversed(scheduler.runtime_decision_log()["records"])
        if request.request_id in record["request_ids"]
        and record["event_kind"] in {"DECISION", "REPLAN"}
    )
    candidates = []
    for candidate in record["candidates"]:
        details = candidate.get("details") or {}
        demands = candidate.get("memory_demands") or []
        sessions = sorted({
            str(demand["demand_id"]).split(":")[1]
            for demand in demands
            if str(demand.get("demand_id", "")).startswith(
                "phone-session:"
            )
        })
        candidates.append({
            "admitted": candidate["admitted"],
            "candidate_id": candidate["route_id"],
            "energy_evidence": details.get("energy_evidence"),
            "executor_id": candidate["executor_id"],
            "fleet_energy_lower_uj": candidate.get(
                "fleet_energy_lower_uj"
            ),
            "fleet_energy_upper_uj": candidate.get(
                "fleet_energy_upper_uj"
            ),
            "maturity": details.get("maturity"),
            "paired_baseline_comparison": details.get(
                "paired_baseline_comparison"
            ),
            "reason": candidate.get("reason"),
            "rejection_reasons": list(
                details.get("rejection_reasons") or ()
            ),
            "residency_break_even": details.get(
                "residency_break_even"
            ),
            "route_family": details.get("route_family"),
            "session_ids": sessions,
            "transition_energy_upper_uj": (
                details.get("cost_breakdown") or {}
            ).get("transition_energy_upper_uj"),
        })
    new_phone_events = plain(
        scheduler.phone_residency_events()[phone_cursor:]
    )
    phone_cursor += len(new_phone_events)
    new_placement_events = plain(
        scheduler.model_placement_events()[placement_cursor:]
    )
    placement_cursor += len(new_placement_events)
    mixed_layouts = [
        layout
        for event in new_phone_events
        for layout in event.get("candidates", [])
        if len({
            shard.get("artifact_sha256")
            for shard in layout.get("shards", [])
        }) > 1
    ]
    requests.append({
        "admission": {
            "attempt_index": ticket.attempt_index,
            "decision_reason": record["decision_reason"],
            "dispatch_state": ticket.dispatch_state,
            "executor_id": ticket.binding.executor_id,
            "selected_route_id": ticket.decision.route_id,
            "ticket_id": ticket.ticket_id,
        },
        "candidate_count": len(candidates),
        "candidate_summaries": candidates,
        "combined_request_index": index,
        "desktop_control_feasible": any(
            candidate["admitted"]
            and candidate["executor_id"].endswith(":desktop")
            for candidate in candidates
        ),
        "mixed_residency_layout_count": len(mixed_layouts),
        "mixed_residency_layouts": mixed_layouts,
        "model_id": model_id,
        "phone_candidate_count": sum(
            bool(candidate["session_ids"]) for candidate in candidates
        ),
        "phone_residency_events": new_phone_events,
        "placement_events": new_placement_events,
        "selected_candidate": next(
            candidate
            for candidate in candidates
            if candidate["candidate_id"] == ticket.decision.route_id
        ),
    })

result = {
    "catalog_sha256": "sha256:"
    + hashlib.sha256(canonical(catalog_value)).hexdigest(),
    "journal_head_sha256": scheduler.runtime_decision_log()[
        "head_record_sha256"
    ],
    "requests": requests,
    "schedule_sha256": replay["schedule_sha256"],
    "schema": "s42-v7-five-request-causal-queue-replay-v6",
    "source_sha256": {
        name: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in {
            "phone_shards.py": repo
            / "research_dev/scheduler/_internal/phone_shards.py",
            "route_generation.py": repo
            / "research_dev/scheduler/_internal/route_generation.py",
            "scheduler.py": repo / "research_dev/scheduler/scheduler.py",
        }.items()
    },
}
output.write_bytes(canonical(result))
summary = []
for row in requests:
    event = (
        row["phone_residency_events"][-1]
        if row["phone_residency_events"]
        else {}
    )
    selected_candidate = row["selected_candidate"]
    authorization = (
        selected_candidate.get("residency_break_even") or {}
    ).get("phone_residency_portfolio_authorization")
    summary.append({
        "index": row["combined_request_index"],
        "model": row["model_id"],
        "route": row["admission"]["selected_route_id"],
        "executor": row["admission"]["executor_id"],
        "decision_reason": row["admission"]["decision_reason"],
        "desktop_feasible": row["desktop_control_feasible"],
        "phone_candidates": row["phone_candidate_count"],
        "mixed_layouts": row["mixed_residency_layout_count"],
        "phone_reason": event.get("reason"),
        "selected_geometry": event.get("selected_geometry_sha256"),
        "selected_objective_uj": event.get("selected_objective_uj"),
        "portfolio_authorization": authorization,
    })
print(json.dumps({
    "output": str(output),
    "requests": summary,
    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
}, sort_keys=True))
