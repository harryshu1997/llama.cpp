"""Run simulate_dispatch with a hook that prints why each affinity refusal happened."""
import sys, traceback, runpy
tree = sys.argv[1]
sys.path.insert(0, tree)
from research_dev.scheduler._internal.runtime_controller import RuntimeController
_orig = RuntimeController.record_dispatch_policy_event
def _rec(self, name, count=1):
    if name == "affinity_refusals":
        et, ev, tb = sys.exc_info()
        frames = [f.name for f in traceback.extract_stack(limit=6)[:-1]]
        print("REFUSAL", et and et.__name__, str(ev)[:160] if ev else "", frames[-3:], flush=True)
    return _orig(self, name, count)
RuntimeController.record_dispatch_policy_event = _rec
sys.argv = ["simulate_dispatch.py"] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
