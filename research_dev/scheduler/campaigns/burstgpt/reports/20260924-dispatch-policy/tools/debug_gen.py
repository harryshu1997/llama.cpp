import os, sys, traceback
target = os.environ["GEN_TARGET"]
sys.argv = ["simulate_dispatch.py"] + sys.argv[1:]
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
import simulate_dispatch as sim
orig_main = sim.main
def patched():
    from research_dev.scheduler._internal import runtime_queue as rq
    armed = {"on": False}
    orig_setattr = rq._DispatchEntry.__setattr__
    def setattr_(self, name, value):
        if armed["on"] and name == "generation" and getattr(self, "decision", None) is not None and self.decision.request_id.endswith(target):
            print("GEN", self.generation if hasattr(self, "generation") else None, "->", value, "state", getattr(self, "state", None))
            print("".join(traceback.format_stack(limit=9)[:-1]))
        orig_setattr(self, name, value)
    rq._DispatchEntry.__setattr__ = setattr_
    orig_replan = None
    from research_dev.scheduler import UnifiedScheduler
    orig_replan = UnifiedScheduler.replan_automated_request
    def replan(self, request_id, **kw):
        if request_id.endswith(target) and kw.get("reason") == "capacity_released_early":
            armed["on"] = True
        try:
            return orig_replan(self, request_id, **kw)
        finally:
            armed["on"] = False
    UnifiedScheduler.replan_automated_request = replan
    orig_main()
sim.main = patched
sim.main()
