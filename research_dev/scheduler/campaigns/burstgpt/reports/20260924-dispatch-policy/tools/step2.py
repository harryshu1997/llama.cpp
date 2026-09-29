import sys, json, time
from replay_common import setup, probe_ready, mk
tree = sys.argv[1]
ctx = setup(tree)
sch = ctx["scheduler"]; req = ctx["req"]; snap = ctx["snap"]; RD = ctx["run_dir"]
from research_dev.scheduler import RuntimeTransitionReceipt
P = "burstgpt_longtail_dev_v2:"
DL = ctx["decision_log"]
def submit(eid, t, idx):
    r, model_id, _ = req(P + eid)
    s = snap(RD / "allsnaps" / f"request-{idx:03d}.json")
    at = max(t, s.captured_at_us)
    ch = sch.observe_automated_runtime_snapshot(s, observed_at_us=at)
    tk = sch.submit_automated_request(r, model_id, s, observed_at_us=at, selection_mode="desktop-baseline")
    print("SUBMIT", eid, tk.decision.route_id[-30:], "start", tk.decision.start_us, "obs changed", ch)
    return tk
def epoch_at(t): return time.monotonic_ns() - t * 1000
submit("000", 1006792, 0)
submit("001", 1998583, 1)
submit("llama-3.2-1b-overlay:00", 3802327, 2)
tk0 = sch.wait_runtime_request(P+"000", epoch_at(1100000))
print("000", tk0.dispatch_state)
submit("002", 79009463, 3)
submit("003", 86205484, 4)
sel = [x for x in DL if x["request_ids"][0]==P+"000" and x["event_kind"]=="ACQUIRED"][0]["selected"]
recs = tuple(mk(RuntimeTransitionReceipt, d) for d in sel["transition_receipts"])
print("receipts", [(r.transition_id, r.started_us, r.finished_us) for r in recs])
sch.record_automated_transition_receipts(P+"000", recs)
s = snap(RD/"allsnaps"/"runtime-000-burstgpt_longtail_dev_v2-000-98100946.json")
ch = sch.observe_automated_runtime_snapshot(s, observed_at_us=max(98100946, s.captured_at_us))
print("observe after load:", ch)
q = sch._runtime_controller.queue
print(json.dumps(q.snapshot()["entry_states"], indent=0))
print(json.dumps(q.snapshot()["causal_predecessors"], indent=0))
import research_dev.scheduler._internal.runtime_residency_projection as rp
t1 = sch.runtime_ticket(P+"001")
print("001 transitions", [(t.transition_id, t.source_state, t.target_state, t.executor_id, t.prepares_device_ids) for t in t1.execution_plan.transitions])
print("obs", [rp.transition_target_is_observed(s, t1, t) for t in t1.execution_plan.transitions])
tickets = {row.request.request_id: row for row in sch._runtime_controller.current_tickets()}
auth = tuple(tickets[r] for r in sch._runtime_controller.projection_request_ids() if r in tickets)
try:
    proj = rp.project_scheduler_residency(s, sch._runtime_capabilities, auth, sch._runtime_manifests, stop_before_ticket_id=t1.ticket_id, causal_predecessors=sch._runtime_controller.projection_causal_predecessors())
    print("proj obs", [rp.transition_target_is_observed(proj, t1, t) for t in t1.execution_plan.transitions])
except Exception as e:
    print("PROJECTION FAILED", type(e).__name__, e, getattr(e, "request_id", None))
submit("004", 99022365, 5)
submit("005", 109433414, 6)
print(json.dumps(q.snapshot()["causal_predecessors"], indent=0))
