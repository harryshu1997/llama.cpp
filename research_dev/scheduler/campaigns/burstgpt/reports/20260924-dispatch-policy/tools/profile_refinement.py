import sys, cProfile, pstats, io, time
from pathlib import Path
tree = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(tree)); sys.path.insert(0, str(tree / "research_dev/scheduler/tests"))
import test_automated_runtime_admission as t
from test_automated_runtime import request, runtime_snapshot
case = t.AutomatedRuntimeAdmissionTests("test_cached_synthetic_refinement_is_below_ten_milliseconds")
case.setUp()
scheduler, manifest = case.scheduler_and_manifest()
snapshot = runtime_snapshot(manifest)
scheduler.generate_automated_candidates(request("warm"), manifest.model_id, snapshot)
N = 60
t0 = time.perf_counter_ns()
for i in range(N):
    scheduler.generate_automated_candidates(request(f"r{i}"), manifest.model_id, snapshot)
print("mean ms %.3f" % ((time.perf_counter_ns() - t0) / N / 1e6))
pr = cProfile.Profile(); pr.enable()
for i in range(20):
    scheduler.generate_automated_candidates(request(f"p{i}"), manifest.model_id, snapshot)
pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(12); print(s.getvalue()[:3500])
case.tearDown()
