"""Run test_dispatch_policy.py against the BASE tree with a read-only API shim.

The shim adds only what the tests need to observe the base dispatcher: the policy value type,
a no-op configure call, and read-only queue views. No dispatch behaviour is added, so every
assertion about work-conserving admission or model affinity sees the legacy dispatcher.
"""
import importlib.util, sys, types, unittest
from pathlib import Path
tree = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(tree))
sys.path.insert(0, str(tree / "research_dev/scheduler/tests"))
import research_dev.scheduler as pkg
spec = importlib.util.spec_from_file_location(
    "research_dev.scheduler._internal.runtime_dispatch_policy",
    Path(__file__).resolve().parents[1] / "newroot/research_dev/scheduler/_internal/runtime_dispatch_policy.py")
policy_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = policy_module
spec.loader.exec_module(policy_module)
pkg.RuntimeDispatchPolicy = policy_module.RuntimeDispatchPolicy
pkg.RuntimeDispatchPolicyError = policy_module.RuntimeDispatchPolicyError
from research_dev.scheduler._internal.runtime_queue import RuntimeDispatchQueue
from research_dev.scheduler._internal.runtime_controller import RuntimeController
from types import MappingProxyType
_init = RuntimeDispatchQueue.__init__
def __init__(self, policy=None):
    _init(self)
RuntimeDispatchQueue.__init__ = __init__
RuntimeDispatchQueue.set_policy = lambda self, policy: None
def dispatch_order_view(self):
    with self._condition:
        return MappingProxyType({rid: MappingProxyType({
            "predecessor_request_ids": tuple(sorted(e.predecessor_request_ids)),
            "residency_transition_barrier": e.residency_transition_barrier,
            "sequence": e.sequence, "state": e.state}) for rid, e in sorted(self._entries.items())})
RuntimeDispatchQueue.dispatch_order_view = dispatch_order_view
RuntimeDispatchQueue.policy_events = lambda self: MappingProxyType({})
RuntimeController.dispatch_order_view = lambda self: self.queue.dispatch_order_view()
from research_dev.scheduler import UnifiedScheduler
UnifiedScheduler.configure_runtime_dispatch_policy = lambda self, policy: None
UnifiedScheduler.runtime_dispatch_policy_state = lambda self: MappingProxyType({
    "bypass_counts": {}, "policy": None, "statistics": {k: 0 for k in (
        "affinity_displaced_attempts", "affinity_displacements", "affinity_refusals",
        "early_capacity_promotions", "publication_replans", "published_work_promotions")}})
import test_dispatch_policy as tests
suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
