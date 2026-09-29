import sys, time, json
sys.path.insert(0, "research_dev/scheduler/tests")
import two_phone_harness as h
from research_dev.scheduler import UnifiedScheduler
from research_dev.scheduler.adapters import interpret_runtime_ticket, llama_server_launch_contract
mode = sys.argv[1] if len(sys.argv) > 1 else "adaptive-decode"
with h.TemporaryModel() as model:
    catalog = h.runtime_catalog(model, declaration=h.co_helpers())
    s = UnifiedScheduler.for_runtime_discovery("enforce")
    s.register_runtime_capabilities(catalog)
    s.register_model_manifest(model)
    snap = h.snapshot(model, catalog)
    value = h.request("r1")
    ticket = s.submit_automated_request(value, model.model_id, snap, selection_mode=mode)
    epoch_ns = time.monotonic_ns() - ticket.decision.start_us * 1_000
    ticket = s.wait_runtime_request("r1", epoch_ns)
    print(ticket.binding.executor_id, ticket.decision.reason, ticket.execution_plan.execution_contract.execution_mode)
    p = ticket.execution_plan.adapter_parameters
    print(sorted(k for k in p if "phone" in k or "dormant" in k))
    print(p.get("dormant_phone_ffn_runtime_v1"))
    cmd = interpret_runtime_ticket(ticket)
    print(type(cmd))
    command = cmd.execution if hasattr(cmd, "execution") else cmd
    c = llama_server_launch_contract(command, model)
    print(json.dumps(dict(c.ffn_environment), indent=1))
