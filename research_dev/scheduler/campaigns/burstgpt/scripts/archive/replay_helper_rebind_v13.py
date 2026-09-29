import json
import os
from pathlib import Path
import sys

from research_dev.scheduler import (
    HeterogeneousRuntimeSnapshot,
    Request,
    RuntimeCapabilityCatalog,
    UnifiedScheduler,
)


repo = Path(sys.argv[1])
gate = Path(sys.argv[2])
physical = Path(sys.argv[3])


def load(path):
    return json.loads(path.read_text(encoding="ascii"))


catalog = RuntimeCapabilityCatalog.from_json(
    load(gate / "inputs/UNIFIED_RUNTIME_CATALOG.json")
)
source_catalog = RuntimeCapabilityCatalog.from_json(
    load(gate / "inputs/OBSERVATION_SOURCE_CATALOG.json")
)
scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
scheduler.register_runtime_capabilities(catalog)
cache = Path(
    "/home/zhihao/.cache/llama.cpp/research-scheduler/gguf-manifests.json"
)
models = {
    "gemma-4-12b-q40-dequant-f16": Path(
        "/home/zhihao/models/gemma-4-12B-Q40-dequant-f16.gguf"
    ),
    "qwen3-14b-q4km-dequant-f16": Path(
        "/home/zhihao/models/Qwen3-14B-Q4KM-dequant-f16.gguf"
    ),
}
for model_id, path in models.items():
    scheduler.register_gguf_model(model_id, path, cache_path=cache)
scheduler.load_automated_observations(
    load(gate / "inputs/AUTOMATED_OBSERVATIONS.json"),
    source_catalog=source_catalog,
)
scheduler.load_adaptive_decode_observations(
    load(gate / "inputs/ADAPTIVE_DECODE_OBSERVATIONS.json"),
    source_catalog=source_catalog,
)

sources = {
    row["combined_request_index"]: row
    for row in load(physical / "run/RESULT.json")["request_results"]
    if row["combined_request_index"] in {49, 43}
}
rows = []
for index in (49, 43):
    if index == 43 and os.environ.get("SIMULATE_FIRST_READY") == "1":
        target = scheduler._model_placement_controller.target_phone_layout()
        ready = scheduler._model_placement_controller.verify_observed_phone_layout(
            target.generation,
            workspace_bytes=target.workspace_bytes,
            verified_at_us=target.proposed_at_us + 1,
            verification_sha256="sha256:" + "1" * 64,
        )
        scheduler._automated_compiler().set_phone_residency_layout(
            ready.layout
        )
    source = sources[index]
    snapshot = HeterogeneousRuntimeSnapshot.from_json(
        load(
            physical / "run/snapshots" /
            ("request-%03d.json" % index)
        )
    )
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
    ticket = scheduler.submit_automated_request(
        request,
        source["model_id"],
        snapshot,
        observed_at_us=snapshot.captured_at_us,
        selection_mode="energy-aware",
    )
    if index == 43 and os.environ.get("SIMULATE_FIRST_READY") == "1":
        scheduler._update_phone_residency_portfolio(
            request,
            scheduler.runtime_model_manifest(source["model_id"]),
            snapshot.captured_at_us + 1,
            snapshot,
        )
    compiler = scheduler._automated_compiler()
    manifest = scheduler.runtime_model_manifest(source["model_id"])
    target = scheduler._model_placement_controller.target_phone_layout()
    rows.append({
        "combined_request_index": index,
        "dispatch_state": ticket.dispatch_state,
        "executor_id": ticket.binding.executor_id,
        "helper_opportunity": (
            None
            if ticket.execution_plan is None
            or ticket.execution_plan.helper_envelope is None
            else ticket.execution_plan.helper_envelope.to_json()
        ),
        "phone_residency_evidence": dict(
            compiler.phone_residency_evidence_status(
                manifest.artifact_sha256
            )
        ),
        "route_id": ticket.decision.route_id,
        "target_layout": None if target is None else target.to_json(),
    })
print(json.dumps({
    "phone_residency_events": [
        dict(row) for row in scheduler.phone_residency_events()
    ],
    "requests": rows,
}, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
