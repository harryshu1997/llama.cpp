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
    print("SUBMIT", eid, tk.decision.route_id, "start", tk.decision.start_us, "obs changed", ch)
    return tk
submit("000", 1006792, 0)
submit("001", 1998583, 1)
submit("llama-3.2-1b-overlay:00", 3802327, 2)
q = sch._runtime_controller.queue
print("probe 000", probe_ready(q, P+"000", 1100000))
print(json.dumps(q.snapshot()["causal_predecessors"], indent=0))
