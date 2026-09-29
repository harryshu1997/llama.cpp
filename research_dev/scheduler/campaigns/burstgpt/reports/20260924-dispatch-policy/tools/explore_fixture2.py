import sys, time
sys.path.insert(0, "../tools")
from explore_fixture import build, show, request, RuntimeDispatchPolicy
for label, policy in (("off", None), ("wc", RuntimeDispatchPolicy(work_conserving_admission=True))):
    print("=====", label)
    sch, a, b, snap, gpu = build(policy)
    a1 = sch.submit_automated_request(request("a1", arrival_us=1_000, output_tokens=640), a.model_id, snap, selection_mode="desktop-baseline"); show(sch, a1)
    b1 = sch.submit_automated_request(request("b1", arrival_us=1_100, output_tokens=8), b.model_id, snap, selection_mode="desktop-baseline"); show(sch, b1)
    a2 = sch.submit_automated_request(request("a2", arrival_us=1_200, output_tokens=4), a.model_id, snap, selection_mode="desktop-baseline"); show(sch, a2)
    print("  a2 finish_upper", a2.decision.finish_upper_us)
    q = sch.runtime_controller_snapshot()["dispatch_queue"]
    print("  preds", q["causal_predecessors"])
    print("  states", {k: (v["state"], v["residency_transition_barrier"]) for k, v in q["entry_states"].items()})
    t1 = sch.wait_runtime_request("a1", time.monotonic_ns() - 1_000 * 1000)
    print("  a1", t1.dispatch_state)
    from replay_common import probe_ready
    print("  a2 probe at 1200:", probe_ready(sch._runtime_controller.queue, "a2", 1_200))
