import sys, time
from dataclasses import replace
sys.path.insert(0, "../tools")
from explore_fixture import build, show, request, RuntimeDispatchPolicy
from test_automated_runtime import FakeAutomatedPhysicalAdapter
from replay_common import probe_ready
for label, policy in (("off", None), ("wc", RuntimeDispatchPolicy(work_conserving_admission=True))):
    print("=====", label)
    sch, a, b, hot, gpu = build(policy)
    cold = replace(hot, residency=())
    a1 = sch.submit_automated_request(request("a1", arrival_us=1_000, output_tokens=640), a.model_id, cold, selection_mode="desktop-baseline"); show(sch, a1)
    a2 = sch.submit_automated_request(request("a2", arrival_us=1_100, output_tokens=8), a.model_id, cold, selection_mode="desktop-baseline"); show(sch, a2)
    t1 = sch.wait_runtime_request("a1", time.monotonic_ns() - a1.decision.start_us * 1000)
    print("  a1", t1.dispatch_state, t1.transition_status)
    receipts = FakeAutomatedPhysicalAdapter._transition_receipts(t1)
    done_at = a1.decision.start_us + 150
    receipts = tuple(replace(r, finished_us=done_at) for r in receipts)
    sch.record_automated_transition_receipts("a1", receipts)
    snap = replace(hot, snapshot_id="hot-after-load", captured_at_us=done_at, valid_until_us=done_at + 10_000_000,
                   memory=replace(hot.memory, snapshot_id="hot-after-load-memory", captured_at_us=done_at, valid_until_us=done_at + 10_000_000))
    print("  observe", sch.observe_automated_runtime_snapshot(snap, observed_at_us=done_at))
    q = sch.runtime_controller_snapshot()["dispatch_queue"]
    print("  states", {k: (v["state"], v["wake_reason"]) for k, v in q["entry_states"].items()}, q["causal_predecessors"])
    r = probe_ready(sch._runtime_controller.queue, "a2", done_at)
    print("  a2 probe", r)
    if hasattr(r, "status") and r.status == "REPLAN_REQUIRED":
        w = sch.wait_runtime_request("a2", time.monotonic_ns() - done_at * 1000)
        n = sch.replan_automated_request("a2", observed_at_us=done_at, reason=w.dispatch_receipt.wake_reason, snapshot=snap,
                                         expected_ticket_id=w.ticket_id, expected_queue_generation=w.dispatch_receipt.queue_generation)
        show(sch, n)
        print("  a2 probe after replan", probe_ready(sch._runtime_controller.queue, "a2", done_at))
    if hasattr(sch, "runtime_dispatch_policy_state"): print("  stats", dict(sch.runtime_dispatch_policy_state())["statistics"])
