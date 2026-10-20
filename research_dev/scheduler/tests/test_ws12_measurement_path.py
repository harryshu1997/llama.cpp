"""WS12: Pixel AOA measurements reach the planner (server shapes, sustained probe, provisioned inventory)."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler._internal.layer_placement import LayerPlacementProfile, solve_placement  # noqa: E402
from research_dev.scheduler._internal.layer_placement import CallKey  # noqa: E402
from research_dev.scheduler._internal.layer_placement_io import problem_from_json  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt import layer_placement_rig as rig  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.tools import layer_placement as tool  # noqa: E402

SHAPE = ('S41SERVERFFNSHAPE {"tokens":1,"columns":17408,"calls":4000,"rpc_mean_ms":8.1,"rpc_p50_ms":7.9,'
         '"compute_mean_ms":7.3,"compute_p50_ms":7.1,"host_mean_ms":0.01,"wait_mean_ms":8.2,"overlap_mean_ms":8.2,'
         '"overlap_p50_ms":8.0}')


def run(*argv) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        code = tool.main(list(argv))
    return code, out.getvalue()


class MeasurementPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_server_shapes_json_folds_into_a_profile(self):
        shapes = self.root / "SERVER_FFN_SHAPES.json"
        shapes.write_text(json.dumps([SHAPE, "unrelated log line"], indent=2))
        code, text = run("shapes", "--model", rig.QWEN, "--n-ff", "17408", "--label",
                         "pixel10pro=pixel10pro-phone:q4k-packed:aoa-bridge", "--default-label", "pixel10pro",
                         str(shapes))
        self.assertEqual(code, 0)
        rows = json.loads(text)
        self.assertEqual([(row["transport"], row["calls"], row["rpc_ms"]) for row in rows], [("aoa-bridge", 4000, 8.1)])
        (self.root / "rows.json").write_text(text)
        base = rig.measured_profile_v1()
        key = CallKey(rig.PIXEL, rig.QWEN, "q4k-packed", 1, "aoa-bridge")
        before = base.calls[key].calls
        code, text = run("profile", "--out", str(self.root / "P.json"), "--add-shapes", str(self.root / "rows.json"))
        self.assertEqual((code, json.loads(text)), (0, {"folded_rows": 1}))
        after = LayerPlacementProfile.from_json(json.loads((self.root / "P.json").read_text()))
        self.assertEqual(after.calls[key].calls, before + 4000)
        # folding again on top of the written profile keeps accumulating (the profile is the learner's store)
        code, _ = run("profile", "--out", str(self.root / "P2.json"), "--base", str(self.root / "P.json"),
                      "--add-shapes", str(self.root / "rows.json"))
        again = LayerPlacementProfile.from_json(json.loads((self.root / "P2.json").read_text()))
        self.assertEqual((code, again.calls[key].calls), (0, before + 8000))

    def test_probe_ingest_reads_pixel_transport_ab_calls(self):
        code, text = run("probe-plan", "--device", rig.PIXEL, "--format", "q4k-packed", "--transport", "aoa-bridge",
                         "--layer-bytes", str(rig.QWEN_Q4K_LAYER), "--layers", "25", "--rows", "1,2")
        self.assertEqual(code, 0)
        plan = json.loads(text)
        self.assertEqual((plan["layers"], plan["rows"]), (25, [1, 2]))
        (self.root / "PLAN.json").write_text(text)
        calls = []
        for rows, compute in ((1, 7000), (2, 9000)):
            for token in range(-2, 4):            # pixel_transport_ab: negative token = warm-up
                for position in range(25):
                    calls.append({"rows": rows, "token": token, "position": position, "layer": position,
                                  "compute_us": compute * (5 if token < 0 else 1), "rpc_us": compute * (5 if token < 0 else 1) + 800,
                                  "overhead_us": 800})
        (self.root / "CALLS.json").write_text(json.dumps(calls))
        (self.root / "P0.json").write_text(json.dumps(rig.measured_profile_v1().to_json()))
        code, text = run("probe-ingest", "--profile", str(self.root / "P0.json"), "--plan", str(self.root / "PLAN.json"),
                         "--model", rig.QWEN, "--calls", str(self.root / "CALLS.json"), "--out", str(self.root / "P1.json"))
        self.assertEqual(code, 0)
        result = json.loads(text)
        self.assertEqual(result["by_rows"]["1"]["calls"], 100.0)          # warm-up tokens skipped
        self.assertAlmostEqual(result["by_rows"]["1"]["rpc_ms"], 7.8)
        self.assertAlmostEqual(result["busy_ms_per_token"], 25 * 7.8)

    def test_inventory_overrides_reach_the_executable_plan(self):
        code, _ = run("inventory", "--out", str(self.root / "I.json"), "--pixel-transport", "aoa-bridge",
                      "--stored", f"{rig.PIXEL}:{rig.QWEN}=13-23", "--qualified", f"{rig.PIXEL}:{rig.QWEN}=13-23",
                      "--stored", f"{rig.OP15}:{rig.QWEN}=0-12", "--qualified", f"{rig.OP15}:{rig.QWEN}=0-12",
                      "--busy-envelope", f"{rig.PIXEL}=120.5")
        self.assertEqual(code, 0)
        inventory = json.loads((self.root / "I.json").read_text())
        pixel = next(row for row in inventory["devices"] if row["device_id"] == rig.PIXEL)
        self.assertEqual((pixel["stored_layers"][rig.QWEN], pixel["busy_envelope_ms"], pixel["route_anchor"],
                          pixel["max_models"]), ("13-23", 120.5, False, 1))
        profile = rig.measured_profile_v1()
        report = tool.placement_report(problem_from_json(inventory, profile))
        owners = {owner: row["layers"] for owner, row in report["executable"]["models"][rig.QWEN]["owners"].items()
                  if row["count"]}
        self.assertEqual(owners, {rig.OP15: "0-12", rig.PIXEL: "13-23"})
        self.assertEqual(run("inventory", "--out", str(self.root / "bad.json"), "--stored", "nope:x=1")[0], 1)
        self.assertEqual(run("inventory", "--out", str(self.root / "bad.json"), "--busy-envelope", "nope=3")[0], 1)
        ideal = solve_placement(problem_from_json(inventory, profile))
        self.assertIn(rig.OP15, {owner for owner in ideal.models[rig.QWEN].owners.values()})


if __name__ == "__main__":
    unittest.main()
