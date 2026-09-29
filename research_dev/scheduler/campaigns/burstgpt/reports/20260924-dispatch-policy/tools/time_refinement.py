import sys, unittest
from pathlib import Path
tree = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(tree)); sys.path.insert(0, str(tree / "research_dev/scheduler/tests"))
import test_automated_runtime_admission as t
values = []
orig = unittest.TestCase.assertLess
def capture(self, a, b, msg=None):
    if b == 10_000_000:
        values.append(a)
    return None
unittest.TestCase.assertLess = capture
for _ in range(int(sys.argv[2])):
    suite = unittest.TestSuite([t.AutomatedRuntimeAdmissionTests("test_cached_synthetic_refinement_is_below_ten_milliseconds")])
    unittest.TextTestRunner(stream=open("/dev/null", "w")).run(suite)
print(Path(sys.argv[1]).name, "mean of per-run means ms: %.2f" % (sum(values) / len(values) / 1e6), "min %.2f max %.2f n=%d" % (min(values)/1e6, max(values)/1e6, len(values)))
