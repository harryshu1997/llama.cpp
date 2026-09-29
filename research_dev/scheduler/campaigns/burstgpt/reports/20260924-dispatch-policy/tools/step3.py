import sys, json, time
from replay_common import setup, probe_ready, mk
tree = sys.argv[1]; policy = sys.argv[2] if len(sys.argv) > 2 else "off"
ctx = setup(tree)
sch = ctx["scheduler"]; req = ctx["req"]; snap = ctx["snap"]; RD = ctx["run_dir"]
if policy != "off":
    from research_dev.scheduler._internal.runtime_dispatch_policy import RuntimeDispatchPolicy
    sch.configure_runtime_dispatch_policy(RuntimeDispatchPolicy(work_conserving_admission=True, model_affinity=(policy=="aff")))
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
submit("002", 79009463, 3)
submit("003", 86205484, 4)
sel = [x for x in DL if x["request_ids"][0]==P+"000" and x["event_kind"]=="ACQUIRED"][0]["selected"]
recs = tuple(mk(RuntimeTransitionReceipt, d) for d in sel["transition_receipts"])
sch.record_automated_transition_receipts(P+"000", recs)
s = snap(RD/"allsnaps"/"runtime-000-burstgpt_longtail_dev_v2-000-98100946.json")
ch = sch.observe_automated_runtime_snapshot(s, observed_at_us=max(98100946, s.captured_at_us))
print("observe after load:", ch)
q = sch._runtime_controller.queue
print({k:(v["state"], v["wake_reason"]) for k,v in q.snapshot()["entry_states"].items()})
print(getattr(sch, "runtime_dispatch_policy_state", lambda: None)())
