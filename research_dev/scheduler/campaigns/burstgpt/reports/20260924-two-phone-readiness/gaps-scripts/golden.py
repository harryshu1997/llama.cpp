import sys, time, json
sys.path.insert(0, "research_dev/scheduler/tests")
import two_phone_harness as h
from research_dev.scheduler import UnifiedScheduler
from research_dev.scheduler._internal.types import canonical_sha256
from research_dev.scheduler._internal.adaptive_decode_planning import adaptive_decode_policies
from research_dev.scheduler.adapters import interpret_runtime_ticket, llama_server_launch_contract
out = {}
with h.TemporaryModel() as model:
    catalog = h.runtime_catalog(model)
    out["catalog"] = canonical_sha256(catalog.to_json())
    s = UnifiedScheduler.for_runtime_discovery("enforce")
    s.register_runtime_capabilities(catalog)
    s.register_model_manifest(model)
    snap = h.snapshot(model, catalog)
    values = s.generate_automated_candidates(h.request("golden"), model.model_id, snap)
    out["candidates"] = canonical_sha256([[row.candidate_id, row.plan.to_json(), list(row.rejection_reasons)] for row in values.candidates])
    base, pols, env = adaptive_decode_policies(values, model, catalog, 30)
    out["policies"] = canonical_sha256([row.to_json() for row in (base, *pols)])
    for mode in ("adaptive-decode", "energy-aware"):
        s = UnifiedScheduler.for_runtime_discovery("enforce")
        s.register_runtime_capabilities(catalog)
        s.register_model_manifest(model)
        rid = "golden-" + mode
        ticket = s.submit_automated_request(h.request(rid), model.model_id, snap, selection_mode=mode)
        ticket = s.wait_runtime_request(rid, time.monotonic_ns() - ticket.decision.start_us * 1000)
        command = interpret_runtime_ticket(ticket)
        contract = llama_server_launch_contract(command, model)
        out["plan-" + mode] = ticket.execution_plan.plan_sha256
        out["launch-" + mode] = canonical_sha256(dict(contract.ffn_environment))
print(json.dumps(out, indent=1, sort_keys=True))
