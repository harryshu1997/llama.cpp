import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
sys.argv = ["simulate_dispatch.py"] + sys.argv[1:]
import simulate_dispatch as sim
def patched(orig=sim.main):
    import research_dev.scheduler._unified.automated_requests_ops.replan as rp
    from research_dev.scheduler._internal import runtime_queue as rq
    orig_fn = rp._replan_priority_compaction_without_followers
    def wrapper(controller, request_id, **kw):
        before = frozenset(controller._runtime_controller.replan_required_requests())
        orig_once = controller._replan_automated_request_once
        calls = {"n": 0}
        def once(rid, **k):
            calls["n"] += 1
            if calls["n"] == 2 or rid != request_id:
                q = controller._runtime_controller.queue
                e = q._entries.get(rid)
                t = controller.runtime_ticket(rid)
                print("FOLLOWER", rid, "queue", (e.state, e.generation, e.wake_reason, sorted(e.predecessor_request_ids)) if e else None,
                      "ticket", t.dispatch_state, t.dispatch_receipt.queue_generation if t.dispatch_receipt else None,
                      "expected", k.get("expected_queue_generation"), "before", rid in before)
            return orig_once(rid, **k)
        controller._replan_automated_request_once = once
        try:
            return orig_fn(controller, request_id, **kw)
        finally:
            del controller._replan_automated_request_once
    rp._replan_priority_compaction_without_followers = wrapper
    orig()
sim.main = patched
sim.main()
