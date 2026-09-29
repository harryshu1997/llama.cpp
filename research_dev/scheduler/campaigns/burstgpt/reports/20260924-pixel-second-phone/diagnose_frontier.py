"""Replay a saved catalog with a diagnostic preference for adaptive split routes."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

from research_dev.scheduler import RuntimeCapabilityCatalog, UnifiedScheduler
from research_dev.scheduler._internal.adaptive_decode_planning import adaptive_decode_policies
from research_dev.scheduler._internal.model_manifest import ModelManifest
from research_dev.scheduler._internal.policy import Request
from research_dev.scheduler._internal.route_generation.costing_rough import RouteRoughCostMixin
from research_dev.scheduler._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("catalog", "snapshot", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    catalog = RuntimeCapabilityCatalog.from_json(json.loads(args.catalog.read_text()))
    model = ModelManifest.from_json(json.loads(args.manifest.read_text()))
    saved = json.loads(args.snapshot.read_text())
    snapshot = HeterogeneousRuntimeSnapshot.from_json(saved["runtime"])
    now = snapshot.captured_at_us
    request = Request(request_id="offline-frontier-diagnosis", workload_id="diagnostic",
                      arrival_us=now, deadline_us=now + 1_000_000_000,
                      input_tokens=318, output_tokens=220, quality_requirement="exact")
    original = RouteRoughCostMixin._rough_visits

    def prefer_split(compiler, *arguments, **keywords):
        rows = original(compiler, *arguments, **keywords)
        patterns = {row.route_key: row for row in compiler._patterns(model)}
        eligible = [row for row in rows if row.required_group is not None
                    and patterns[row.route_key].resident_envelope
                    and patterns[row.route_key].route_family == "operator_split"]
        groups = {row.required_group for row in eligible}
        selected = {min((row for row in eligible if row.required_group == group),
                        key=lambda row: (not row.rough_memory_feasible, row.rough_energy_uj,
                                         row.rough_latency_us, row.visit_id)).visit_id
                    for group in groups}
        return tuple(replace(row, mandatory=row.visit_id in selected)
                     if row.required_group in groups and (row.mandatory or row.visit_id in selected)
                     else row for row in rows)

    result = {"note": "Offline counterfactual only; no physical execution or admission bypass. "
                      "Fresh scheduler with saved physical snapshot, not a full queue replay."}
    for name, implementation in (("current", original), ("prefer_split_diagnostic", prefer_split)):
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(catalog)
        scheduler.register_model_manifest(model)
        with patch.object(RouteRoughCostMixin, "_rough_visits", implementation):
            candidates = scheduler.generate_automated_candidates(
                request, model.model_id, snapshot, observed_at_us=now)
            _baseline, policies, envelope = adaptive_decode_policies(
                candidates, model, catalog, request.output_tokens)
            result[name] = {
                "candidate_count": len(candidates.candidates),
                "ffn_candidates": [{"id": row.candidate_id, "family": row.route_family,
                                    "reasons": row.rejection_reasons}
                                   for row in candidates.candidates if row.assisted_operator_kind == "ffn"],
                "adaptive_policies": [{"columns": row.columns, "layer_mask": row.layer_mask,
                                       "device_layer_masks": row.device_layer_masks} for row in policies],
                "envelope": getattr(envelope, "candidate_id", None),
            }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({name: len(result[name]["adaptive_policies"])
                      for name in ("current", "prefer_split_diagnostic")}))


if __name__ == "__main__":
    main()
