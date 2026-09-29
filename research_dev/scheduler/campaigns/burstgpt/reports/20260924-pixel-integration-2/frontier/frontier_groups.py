"""Print rough visits by required group with family/energy/mandatory and frontier selection."""
import json, sys
from pathlib import Path
from unittest.mock import patch
from research_dev.scheduler import RuntimeCapabilityCatalog, UnifiedScheduler
from research_dev.scheduler._internal.adaptive_decode_planning import adaptive_decode_policies
from research_dev.scheduler._internal.model_manifest import ModelManifest
from research_dev.scheduler._internal.policy import Request
from research_dev.scheduler._internal.route_generation.costing_rough import RouteRoughCostMixin
from research_dev.scheduler._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot

catalog = RuntimeCapabilityCatalog.from_json(json.loads(Path(sys.argv[1]).read_text()))
saved = json.loads(Path(sys.argv[2]).read_text())
model = ModelManifest.from_json(json.loads(Path(sys.argv[3]).read_text()))
snapshot = HeterogeneousRuntimeSnapshot.from_json(saved.get("runtime", saved))
now = snapshot.captured_at_us
request = Request(request_id="frontier-groups", workload_id="diagnostic", arrival_us=now,
                  deadline_us=now + 1_000_000_000, input_tokens=int(sys.argv[4]) if len(sys.argv) > 4 else 318,
                  output_tokens=int(sys.argv[5]) if len(sys.argv) > 5 else 220, quality_requirement="exact")
captured = {}
original = RouteRoughCostMixin._rough_visits
def spy(compiler, *a, **k):
    rows = original(compiler, *a, **k)
    captured["rows"] = rows
    captured["patterns"] = {p.route_key: p for p in compiler._patterns(model)}
    return rows
scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
scheduler.register_runtime_capabilities(catalog)
scheduler.register_model_manifest(model)
with patch.object(RouteRoughCostMixin, "_rough_visits", spy):
    candidates = scheduler.generate_automated_candidates(request, model.model_id, snapshot, observed_at_us=now)
_b, policies, envelope = adaptive_decode_policies(candidates, model, catalog, request.output_tokens)
selected = {row.candidate_id for row in candidates.candidates}
groups = {}
for row in captured["rows"]:
    groups.setdefault(row.required_group, []).append(row)
for group, rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
    print("GROUP", group)
    for row in sorted(rows, key=lambda r: r.rough_energy_uj):
        p = captured["patterns"].get(row.route_key)
        print("   %-5s mand=%d feas=%d E=%12d L=%12d %s" % (p.route_family[:12] if p else "?", row.mandatory, row.rough_memory_feasible,
              row.rough_energy_uj, row.rough_latency_us, row.route_key[:110]))
print("candidates", len(candidates.candidates), "ffn", sum(c.assisted_operator_kind == "ffn" for c in candidates.candidates),
      "split", sum(c.route_family == "operator_split" for c in candidates.candidates))
print("policies", [(p.columns, p.layer_mask, p.device_layer_masks) for p in policies])
