import hashlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path

from research_dev.scheduler import (
    HeterogeneousRuntimeSnapshot,
    Request,
    RuntimeCapabilityCatalog,
    UnifiedScheduler,
)
from research_dev.scheduler.campaigns.burstgpt.runner import canonical


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def plain(value):
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


home = Path("/home/zhihao")
repo = Path(os.environ.get(
    "S42_REPLAY_REPO_ROOT",
    str(home / "llama.cpp-release"),
))
source_root = Path(os.environ.get(
    "S42_REPLAY_SOURCE_ROOT",
    str(home / "s42-work-conserving-mixed-gate-20260830-v1"),
))
catalog_root = Path(os.environ.get(
    "S42_REPLAY_CATALOG_ROOT",
    str(source_root),
))
physical_root = Path(os.environ.get(
    "S42_REPLAY_PHYSICAL_ROOT",
    str(source_root),
))
result_root = Path(os.environ.get(
    "S42_REPLAY_RESULT_ROOT",
    str(physical_root),
))
snapshot_root = Path(os.environ.get(
    "S42_REPLAY_SNAPSHOT_ROOT",
    str(physical_root),
))
output_root = Path(os.environ.get(
    "S42_REPLAY_OUTPUT_ROOT",
    str(home / "s42-dimensional-residency-replay-20260830-v8"),
))
if output_root.exists():
    raise SystemExit("refusing to overwrite " + str(output_root))
output_root.mkdir()

catalog_path = catalog_root / "inputs/UNIFIED_RUNTIME_CATALOG.json"
source_catalog_path = catalog_root / "inputs/OBSERVATION_SOURCE_CATALOG.json"
automated_path = catalog_root / "inputs/AUTOMATED_OBSERVATIONS.json"
adaptive_path = catalog_root / "inputs/ADAPTIVE_DECODE_OBSERVATIONS.json"
physical_result = load(result_root / "run/RESULT.json")
catalog = RuntimeCapabilityCatalog.from_json(load(catalog_path))
source_catalog = RuntimeCapabilityCatalog.from_json(load(source_catalog_path))
scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
scheduler.register_runtime_capabilities(catalog)

model_paths = {
    "gemma-4-12b-q40-dequant-f16": (
        home / "models/gemma-4-12B-Q40-dequant-f16.gguf"
    ),
    "qwen3-14b-q4km-dequant-f16": (
        home / "models/Qwen3-14B-Q4KM-dequant-f16.gguf"
    ),
}
cache = home / ".cache/llama.cpp/research-scheduler/gguf-manifests.json"
registration_timings_us = []
for model_id, model_path in model_paths.items():
    started_ns = time.perf_counter_ns()
    scheduler.register_gguf_model(model_id, model_path, cache_path=cache)
    registration_timings_us.append(
        (model_id, (time.perf_counter_ns() - started_ns) // 1000)
    )
scheduler.load_automated_observations(
    load(automated_path), source_catalog=source_catalog
)
scheduler.load_adaptive_decode_observations(
    load(adaptive_path), source_catalog=source_catalog
)

request_rows = physical_result["request_results"]
expected_indices = [36, 41, 42, 43, 44]
if [row["combined_request_index"] for row in request_rows] != expected_indices:
    raise SystemExit("preserved five-request identity differs")

arrived_output_tokens = {}
event_cursor = 0
replay_rows = []
for source in request_rows:
    index = source["combined_request_index"]
    request = Request(
        request_id=source["request_id"],
        workload_id="physical:" + source["model_id"],
        arrival_us=source["replay_arrival_us"],
        deadline_us=(
            source["replay_arrival_us"] + source["source_slo_us"]
        ),
        input_tokens=source["input_tokens"],
        output_tokens=source["output_tokens"],
        quality_requirement="semantic",
    )
    snapshot = HeterogeneousRuntimeSnapshot.from_json(load(
        snapshot_root / f"run/snapshots/request-{index:03d}.json"
    ))
    submission_started_ns = time.perf_counter_ns()
    ticket = scheduler.submit_automated_request(
        request,
        source["model_id"],
        snapshot,
        observed_at_us=snapshot.captured_at_us,
        selection_mode="energy-aware",
    )
    submission_duration_us = (
        time.perf_counter_ns() - submission_started_ns
    ) // 1000
    artifact = ticket.model.artifact_sha256
    arrived_output_tokens[artifact] = (
        arrived_output_tokens.get(artifact, 0) + request.output_tokens
    )
    events = scheduler.phone_residency_events()
    new_events = plain(events[event_cursor:])
    event_cursor = len(events)
    evaluations = [
        event for event in new_events if event["kind"] == "EVALUATED"
    ]
    latest = evaluations[-1]
    if latest["queue_work_by_artifact"] != dict(sorted(
        arrived_output_tokens.items()
    )):
        raise SystemExit(
            "arrived token work differs at request " + request.request_id
        )
    if latest.get("queue_work_unit") != "decode_token":
        raise SystemExit("phone residency work unit is not decode tokens")
    selected_geometry = latest.get("selected_geometry_sha256")
    selected_layout = next(
        (
            layout for layout in latest["candidates"]
            if layout["geometry_sha256"] == selected_geometry
        ),
        None,
    )
    planning_artifacts = sorted({
        shard["artifact_sha256"]
        for shard in (() if selected_layout is None else selected_layout["shards"])
    })
    dimensional_rows = []
    for layout in latest["candidates"]:
        for layout_artifact, benefit in (
            layout["queue_benefit_by_artifact"].items()
        ):
            work = layout["queued_work_by_artifact"][layout_artifact]
            if benefit % work:
                raise SystemExit("layout benefit is not token normalized")
            dimensional_rows.append({
                "artifact_sha256": layout_artifact,
                "benefit_uJ_per_token": benefit // work,
                "queue_benefit_uJ": benefit,
                "remaining_tokens": work,
            })
    plan = ticket.execution_plan
    helper = None if plan is None else plan.helper_envelope
    base_resource_ids = set(() if plan is None else plan.resource_ids)
    helper_plan_resource_ids = set(
        () if helper is None else helper.helper_plan.resource_ids
    )
    helper_resource_ids = helper_plan_resource_ids - base_resource_ids
    committed_resource_ids = {
        lease.resource_id for lease in ticket.decision.leases
    }
    if committed_resource_ids & helper_resource_ids:
        raise SystemExit("desktop base reserves future helper resources")
    replay_rows.append({
        "artifact_sha256": artifact,
        "base_execution_mode": (
            None if plan is None else plan.execution_contract.execution_mode
        ),
        "base_resource_ids": sorted(base_resource_ids),
        "combined_request_index": index,
        "decision_reason": next(
            row["decision_reason"]
            for row in reversed(
                scheduler.runtime_decision_log()["records"]
            )
            if request.request_id in row["request_ids"]
            and row["event_kind"] in {"DECISION", "REPLAN"}
        ),
        "dimensional_layout_rows": dimensional_rows,
        "dispatch_state": ticket.dispatch_state,
        "dormant_phone_ffn_runtime": bool(
            plan is not None
            and plan.adapter_parameters.get(
                "dormant_phone_ffn_runtime_v1"
            )
        ),
        "dormant_phone_ffn_runtime_sha256": (
            None
            if plan is None
            or not plan.adapter_parameters.get(
                "dormant_phone_ffn_runtime_v1"
            )
            else "sha256:" + hashlib.sha256(
                plan.adapter_parameters[
                    "dormant_phone_ffn_runtime_v1"
                ].encode("ascii")
            ).hexdigest()
        ),
        "helper_envelope": None if helper is None else {
            "phone_layout_generation": helper.phone_layout_generation,
            "phone_layout_geometry_sha256": (
                helper.phone_layout_geometry_sha256
            ),
            "resource_ids": sorted(helper_resource_ids),
        },
        "phone_residency_events": new_events,
        "planning_artifacts": planning_artifacts,
        "planned_start_us": ticket.decision.start_us,
        "request_id": request.request_id,
        "submission_duration_us": submission_duration_us,
        "selected_executor_id": ticket.binding.executor_id,
        "selected_route_id": ticket.decision.route_id,
    })

all_events = plain(scheduler.phone_residency_events())
mixed = [
    layout
    for event in all_events
    if event["kind"] == "EVALUATED"
    for layout in event.get("candidates", [])
    if len({
        shard["artifact_sha256"] for shard in layout["shards"]
    }) > 1
]
generations = [
    event.get("selected_layout_generation")
    for event in all_events
    if event["kind"] == "EVALUATED"
    and event.get("selected_layout_generation") is not None
]
if not mixed:
    raise SystemExit("mixed Qwen/Gemma layouts were not evaluated")
if not generations or max(generations) < 1:
    raise SystemExit("phone layout generation did not advance")
if not any(row["helper_envelope"] is not None for row in replay_rows):
    raise SystemExit("no nonblocking phone helper envelope was published")
missing_dormant = [
    {
        "artifact_sha256": row["artifact_sha256"],
        "combined_request_index": row["combined_request_index"],
        "helper_envelope": row["helper_envelope"],
        "request_id": row["request_id"],
    }
    for row in replay_rows
    if (
        row["artifact_sha256"] in row["planning_artifacts"]
        and not row["dormant_phone_ffn_runtime"]
    )
]
if missing_dormant:
    raise SystemExit(
        "desktop launch lacks dormant phone FFN runtime: "
        + json.dumps(missing_dormant, sort_keys=True)
    )
runtime_hashes_by_artifact = {}
for row in replay_rows:
    if row["dormant_phone_ffn_runtime_sha256"] is None:
        continue
    runtime_hashes_by_artifact.setdefault(
        row["artifact_sha256"], set()
    ).add(row["dormant_phone_ffn_runtime_sha256"])
if any(len(values) != 1 for values in runtime_hashes_by_artifact.values()):
    raise SystemExit("dormant phone runtime changes across one model")
if any(row["base_execution_mode"] != "desktop" for row in replay_rows):
    raise SystemExit("a cold phone route delayed immediate desktop dispatch")

result = {
    "assertions": {
        "desktop_base_does_not_reserve_phone": True,
        "desktop_dispatch_while_phone_prepares": True,
        "desktop_launch_is_helper_attachable": True,
        "dormant_runtime_is_stable_per_artifact": True,
        "future_arrivals_not_visible": True,
        "mixed_layouts_evaluated": True,
        "no_request_waits_for_phone_readiness": True,
        "phone_layout_generation_at_least_one": True,
        "token_weighted_objective": True,
    },
    "input_hashes": {
        path.name: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            catalog_path,
            source_catalog_path,
            automated_path,
            adaptive_path,
            result_root / "run/RESULT.json",
        )
    },
    "maximum_phone_layout_generation": max(generations),
    "mixed_layout_count": len(mixed),
    "phone_residency_events": all_events,
    "runtime_decision_log": plain(scheduler.runtime_decision_log()),
    "runtime_decision_timings": plain(
        scheduler.runtime_decision_timings()
    ),
    "model_placement_controller_stats": plain(
        scheduler.model_placement_controller_stats()
    ),
    "requests": replay_rows,
    "registration_timings_us": [
        {"model_id": model_id, "duration_us": duration_us}
        for model_id, duration_us in registration_timings_us
    ],
    "schema": "s42-dimensional-phone-residency-replay-v1",
    "source_hashes": {
        name: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in {
            "model_placement_controller.py": repo / (
                "research_dev/scheduler/_internal/"
                "model_placement_controller.py"
            ),
            "phone_shards.py": repo / (
                "research_dev/scheduler/_internal/phone_shards.py"
            ),
            "route_generation.py": repo / (
                "research_dev/scheduler/_internal/route_generation.py"
            ),
            "scheduler.py": repo / "research_dev/scheduler/scheduler.py",
            "llama_server.py": repo / (
                "research_dev/scheduler/adapters/llama_server.py"
            ),
            "runtime.py": repo / (
                "research_dev/scheduler/adapters/runtime.py"
            ),
        }.items()
    },
}
result_path = output_root / "RESULT.json"
result_path.write_bytes(canonical(result))
print(json.dumps({
    "maximum_phone_layout_generation": max(generations),
    "mixed_layout_count": len(mixed),
    "output": str(result_path),
    "result_sha256": "sha256:"
        + hashlib.sha256(result_path.read_bytes()).hexdigest(),
    "requests": [
        {
            "decision_reason": row["decision_reason"],
            "helper_generation": (
                None if row["helper_envelope"] is None
                else row["helper_envelope"]["phone_layout_generation"]
            ),
            "index": row["combined_request_index"],
            "route": row["selected_route_id"],
            "submission_duration_us": row["submission_duration_us"],
        }
        for row in replay_rows
    ],
}, sort_keys=True))
