import sys
sys.path.insert(0, "research_dev/scheduler/tests")
import two_phone_harness as h
from research_dev.scheduler import UnifiedScheduler
from research_dev.scheduler._internal.adaptive_decode_planning import adaptive_decode_policies
with h.TemporaryModel() as model:
    catalog = h.runtime_catalog(model, declaration=h.co_helpers())
    for row in catalog.composite_executors:
        print(row.executor_id, row.participant_device_ids, row.adapter_parameters.get("ffn_column_quantum"), len(row.operator_ids))
    s = UnifiedScheduler.for_runtime_discovery("enforce")
    s.register_runtime_capabilities(catalog)
    s.register_model_manifest(model)
    snap = h.snapshot(model, catalog)
    values = s.generate_automated_candidates(h.request("r1"), model.model_id, snap)
    print("baseline", values.baseline_route_id)
    for row in values.candidates:
        p = row.plan.adapter_parameters
        print(row.candidate_id, row.device_ids, row.plan.execution_contract.execution_mode, row.rejection_reasons, p.get("ffn_resident_layer_mask"), p.get("ffn_column_quantum"), "phone_helpers" in p)
    base, pols, env = adaptive_decode_policies(values, model, catalog, 30)
    print(env and env.candidate_id)
    for pol in pols:
        print(pol.layer_mask, pol.columns, pol.split_fraction_ppm, pol.device_layer_masks)
