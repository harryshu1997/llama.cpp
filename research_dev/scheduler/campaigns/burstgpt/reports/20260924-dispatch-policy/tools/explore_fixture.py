import sys, time, tempfile
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, "research_dev/scheduler/tests")
from test_automated_runtime import catalog, executor_state, request, runtime_snapshot, write_synthetic_gguf
from research_dev.scheduler import RuntimeCapabilityCatalog, UnifiedScheduler, RuntimeDispatchPolicy

tmp = tempfile.TemporaryDirectory()
def build(policy=None, lanes=2):
    source = catalog()
    gpu = replace(source.executor_by_device["accelerator-b"], qualified_fallback=True,
                  exclusive_residency_resource_id="compute:accelerator-b")
    transition = replace(next(r for r in source.transitions if r.device_id == "accelerator-b"),
                         executor_id=gpu.executor_id, prepares_device_ids=("accelerator-b",),
                         resource_slots={"compute:accelerator-b": lanes})
    resources = dict(source.resources)
    for rid in ("compute:accelerator-b", "link:pcie-in", "link:pcie-out"):
        resources[rid] = replace(resources[rid], capacity=lanes)
    profile = RuntimeCapabilityCatalog.from_json(replace(source, executors=(gpu,), composite_executors=(),
        transitions=(transition,), resources=resources).to_json())
    sch = UnifiedScheduler.for_runtime_discovery("enforce")
    if policy is not None:
        sch.configure_runtime_dispatch_policy(policy)
    sch.register_runtime_capabilities(profile)
    pa = Path(tmp.name) / "a.gguf"; pb = Path(tmp.name) / "b.gguf"
    if not pa.exists():
        write_synthetic_gguf(pa, block_count=2, sliding_window=32); write_synthetic_gguf(pb)
    a = sch.register_gguf_model("model-a", pa); b = sch.register_gguf_model("model-b", pb)
    snap = runtime_snapshot(a, include_phone=False, resident_devices=("accelerator-b",), gpu_free_slots=lanes)
    snap = replace(snap, executors={gpu.executor_id: executor_state(gpu.executor_id, free_slots=lanes)},
                   residency=tuple(replace(r, executor_id=gpu.executor_id) for r in snap.residency))
    return sch, a, b, snap, gpu

def show(sch, t):
    print("  ", t.request.request_id, "start", t.decision.start_us, "end", max(l.reserved_until_us for l in t.decision.leases),
          "transitions", [(x.source_state, x.target_state, len(x.evictions)) for x in t.execution_plan.transitions],
          "lanes", [(l.resource_id, l.lanes, l.start_us, l.reserved_until_us) for l in t.decision.leases if l.resource_id.startswith("compute")])

if __name__ == "__main__":
  for label, policy in (("off", None), ("wc", RuntimeDispatchPolicy(work_conserving_admission=True)), ("aff", RuntimeDispatchPolicy(work_conserving_admission=True, model_affinity=True, affinity_maximum_bypasses=1))):
      print("=====", label)
      sch, a, b, snap, gpu = build(policy)
      a1 = sch.submit_automated_request(request("a1", arrival_us=1_000, output_tokens=64), a.model_id, snap, selection_mode="desktop-baseline"); show(sch, a1)
      b1 = sch.submit_automated_request(request("b1", arrival_us=1_100, output_tokens=8), b.model_id, snap, selection_mode="desktop-baseline"); show(sch, b1)
      a2 = sch.submit_automated_request(request("a2", arrival_us=1_200, output_tokens=4), a.model_id, snap, selection_mode="desktop-baseline"); show(sch, a2)
      a3 = sch.submit_automated_request(request("a3", arrival_us=1_300, output_tokens=64), a.model_id, snap, selection_mode="desktop-baseline"); show(sch, a3)
      a4 = sch.submit_automated_request(request("a4", arrival_us=1_400, output_tokens=64), a.model_id, snap, selection_mode="desktop-baseline"); show(sch, a4)
      q = sch.runtime_controller_snapshot()["dispatch_queue"]
      print("  preds", q["causal_predecessors"])
      print("  states", {k: (v["state"], v["residency_transition_barrier"], v["wake_reason"]) for k, v in q["entry_states"].items()})
      if hasattr(sch, "runtime_dispatch_policy_state"): print("  stats", dict(sch.runtime_dispatch_policy_state())["statistics"], dict(sch.runtime_dispatch_policy_state())["bypass_counts"])
