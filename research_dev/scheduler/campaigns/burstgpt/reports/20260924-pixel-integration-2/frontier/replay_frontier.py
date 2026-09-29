"""Replay saved snapshots through candidate generation; print candidate ids and adaptive policies as JSON."""
import json, sys
from pathlib import Path
from research_dev.scheduler import RuntimeCapabilityCatalog, UnifiedScheduler
from research_dev.scheduler._internal.adaptive_decode_planning import adaptive_decode_policies
from research_dev.scheduler._internal.model_manifest import ModelManifest
from research_dev.scheduler._internal.policy import Request
from research_dev.scheduler._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot

catalog = RuntimeCapabilityCatalog.from_json(json.loads(Path(sys.argv[1]).read_text()))
manifests = [ModelManifest.from_json(json.loads(Path(p).read_text())) for p in sys.argv[2].split(",")]
out = {}
for snap_path in sys.argv[3:]:
    saved = json.loads(Path(snap_path).read_text())
    snapshot = HeterogeneousRuntimeSnapshot.from_json(saved.get("runtime", saved))
    now = snapshot.captured_at_us
    for model in manifests:
        if model.artifact_sha256 not in {c.artifact_sha256 for c in catalog.composite_executors if c.artifact_sha256}:
            continue
        request = Request(request_id="replay", workload_id="diagnostic", arrival_us=now, deadline_us=now + 10**9,
                          input_tokens=318, output_tokens=220, quality_requirement="exact")
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(catalog)
        scheduler.register_model_manifest(model)
        try:
            values = scheduler.generate_automated_candidates(request, model.model_id, snapshot, observed_at_us=now)
            _b, policies, envelope = adaptive_decode_policies(values, model, catalog, 220)
            out[Path(snap_path).name + ":" + model.model_id] = {
                "candidates": sorted(c.candidate_id for c in values.candidates),
                "admitted": sorted(c.candidate_id for c in values.candidates if c.admitted),
                "policies": [[p.columns, p.layer_mask] for p in policies],
                "envelope": getattr(envelope, "candidate_id", None)}
        except Exception as error:
            out[Path(snap_path).name + ":" + model.model_id] = {"error": repr(error)[:300]}
print(json.dumps(out, indent=1, sort_keys=True))
