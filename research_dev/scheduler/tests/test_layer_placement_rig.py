"""WS11 on the recorded rig: measured profile, today's placement, replay of block-1/2 servers, shadow mode."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler._internal.layer_placement import (  # noqa: E402
    LayerPlacementProfile,
    layer_mask,
    solve_placement,
    with_envelope_caps,
)
from research_dev.scheduler._internal.layer_placement_control import MeasuredPlacementConfig  # noqa: E402
from research_dev.scheduler._internal.layer_placement_io import (  # noqa: E402
    inventory_to_json,
    parse_shape_line,
    problem_from_json,
)
from research_dev.scheduler.campaigns.burstgpt import layer_placement_rig as rig  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.layer_placement_shadow import run_shadow  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.tools import layer_placement as tool  # noqa: E402


PROFILE = rig.measured_profile_v1()


def owners(plan, model):
    return {device: value["layers"] for device, value in plan.models[model].to_json()["owners"].items()
            if value["count"]}


class RecordedDataTests(unittest.TestCase):
    def test_fixture_and_measured_means(self):
        servers = rig.recorded_servers()
        self.assertEqual(sum(len(row["shape_lines"]) for row in servers), 139)
        self.assertTrue(all(parse_shape_line(line) for row in servers for line in row["shape_lines"]))
        calls = PROFILE.calls
        gemma = calls[next(key for key in calls if (key.device_id, key.model_id, key.rows) == (rig.OP15, rig.GEMMA, 1))]
        self.assertAlmostEqual(gemma.rpc_ms, 6.99, delta=0.01)
        self.assertEqual(gemma.calls, 230_848)
        pixel = [key for key in calls if key.device_id == rig.PIXEL and key.rows == 1]
        self.assertEqual({key.transport for key in pixel}, {"adb-tcp", "aoa-bridge"})

    def test_desktop_fit_and_provenance(self):
        power = PROFILE.desktop_power
        self.assertEqual(power.provenance, "derived")
        self.assertTrue(140 < power.ffn_w < 160 and 40 < power.wait_w < 70 and 10 < power.other_w < 40, power)
        self.assertEqual(PROFILE.desktop_ffn_ms[(rig.QWEN, 1)].provenance, "measured")
        self.assertAlmostEqual(PROFILE.desktop_ffn_ms[(rig.QWEN, 1)].value, 18.696)
        self.assertTrue(12.5 < PROFILE.desktop_ffn_ms[(rig.GEMMA, 1)].value < 14.0)
        self.assertEqual({key[0] for key in PROFILE.energy_per_call}, {rig.OP15, rig.PIXEL})
        again = LayerPlacementProfile.from_json(json.loads(json.dumps(PROFILE.to_json())))
        self.assertEqual(again.to_json(), PROFILE.to_json())

    def test_busy_envelopes_and_rows_mix(self):
        envelope = rig.busy_envelopes_v1()
        self.assertTrue(170 < envelope[rig.OP15] < 200 and 75 < envelope[rig.PIXEL] < 90, envelope)
        mix = rig.rows_mix_v1()
        self.assertAlmostEqual(mix[rig.QWEN][1], 0.80, delta=0.02)
        self.assertAlmostEqual(mix[rig.GEMMA][2], 0.33, delta=0.02)


class TodayPlacementTests(unittest.TestCase):
    def problem(self, **options):
        return problem_from_json(rig.rig_inventory_v1(profile=PROFILE, **options), PROFILE)

    def test_measured_placement_on_todays_transport(self):
        plan = solve_placement(self.problem(pixel_transport="adb-tcp"))
        self.assertEqual(plan.solver, "exact")
        self.assertEqual(owners(plan, rig.QWEN), {rig.OP15: "0-17", rig.PIXEL: "18-24"})
        self.assertEqual(owners(plan, rig.GEMMA), {rig.OP15: "0-26"})   # the OP15 has room: 9 per session
        self.assertEqual(plan.priors_used, ())
        for model in (rig.QWEN, rig.GEMMA):
            row = plan.models[model]
            self.assertGreater(row.current_energy_j_per_token, row.energy_j_per_token)
            self.assertLess(row.chain_ms_by_rows[1], row.cpu_only_chain_ms_by_rows[1])
        self.assertLessEqual(plan.device_usage[rig.OP15][rig.GEMMA], 3 * rig.OP15_SESSION_LIMIT)

    def test_aoa_variants(self):
        expected = solve_placement(self.problem(pixel_transport="aoa-bridge"))
        self.assertEqual(owners(expected, rig.QWEN), {rig.PIXEL: "0-24"})
        risky = replace(self.problem(pixel_transport="aoa-bridge"), uncertainty_ppm=tool.RISK,
                        confident_calls=tool.CONFIDENT_CALLS)
        self.assertEqual(owners(solve_placement(risky), rig.QWEN), {rig.OP15: "0-17", rig.PIXEL: "18-24"})
        safe = with_envelope_caps(self.problem(pixel_transport="aoa-bridge"), 250_000)
        self.assertEqual(owners(solve_placement(safe), rig.QWEN), {rig.OP15: "0-12", rig.PIXEL: "13-24"})

    def test_user_proposal_when_the_op15_gemma_layout_is_held(self):
        problem = self.problem(pixel_transport="adb-tcp")
        problem = replace(problem, devices=tuple(
            replace(device, allowed_layers=MappingProxyType({rig.QWEN: layer_mask(range(25)),
                                                             rig.GEMMA: layer_mask(range(24))}))
            if device.device_id == rig.OP15 else device for device in problem.devices))
        plan = solve_placement(problem)
        self.assertEqual(owners(plan, rig.GEMMA), {rig.OP15: "0-23", rig.PIXEL: "24-26"})
        self.assertTrue(any(":gemma-4-12b-q40-dequant-f16:q4_0-packed:" in row for row in plan.priors_used))

    def test_conservative_capacity_and_robustness(self):
        low = solve_placement(self.problem(op15_capacity=rig.OP15_LOWEST_LIVE_LIMIT))
        self.assertEqual(owners(low, rig.GEMMA), {rig.OP15: "0-25", rig.PIXEL: "26"})
        gone = solve_placement(self.problem(available={rig.OP15: False}))
        self.assertEqual(owners(gone, rig.QWEN), {rig.PIXEL: "0-24"})
        no_pixel = solve_placement(self.problem(available={rig.PIXEL: False}))
        self.assertEqual(owners(no_pixel, rig.QWEN), {rig.OP15: "0-17", "desktop-cpu": "18-24"})

    def test_inventory_round_trip(self):
        problem = self.problem()
        again = problem_from_json(inventory_to_json(problem), PROFILE)
        self.assertEqual(inventory_to_json(again), inventory_to_json(problem))


class ReplayTests(unittest.TestCase):
    def test_replay_is_fail_closed_and_reacts_to_a_lost_device(self):
        record = tool.replay()
        self.assertEqual(record["errors"], [])
        self.assertEqual(record["decision_counts"], {"BLOCKED_QUALIFICATION": 3})
        self.assertEqual(record["target"][rig.QWEN], {"desktop-cpu": "24", rig.OP15: "0-17", rig.PIXEL: "18-23"})
        needs = [row["details"]["needs"] for row in record["decisions"] if row["model_id"] == rig.GEMMA]
        self.assertTrue(needs and needs[-1][0]["layers"] == "24-26", needs)
        dropped = tool.replay(drop={rig.PIXEL: 3000.0})
        self.assertIn("MASK_OUT_NOW", dropped["decision_counts"])
        self.assertEqual(dropped["target"][rig.QWEN], {"desktop-cpu": "18-24", rig.OP15: "0-17"})


class ShadowTests(unittest.TestCase):
    def test_shadow_record_from_a_run_directory(self):
        servers = [row for row in rig.recorded_servers() if row["run"] == "p0m3"]
        paths = {rig.QWEN: "/models/qwen.gguf", rig.GEMMA: "/models/gemma.gguf"}
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            for index, server in enumerate(servers):
                lines = [f"0.00.1 I llama_model_loader: loaded meta data with 28 key-value pairs and 443 tensors from "
                         f"{paths[server['model']]} (version GGUF V3 (latest))"]
                lines += [f"S41SERVERFFNHELPER label={row['label']} layer_mask={row['layer_mask']} transport=x host=y "
                          "port=0 connection=deferred" for row in server["helpers"]]
                lines += server["shape_lines"]
                (run / f"large-model-{index + 2}-physical-hot-desktop.stderr").write_text("\n".join(lines) + "\n")
            (run / "large-model-1-physical-cold-desktop.stderr").write_text("no model here\n")
            result = {"helper_membership_events": [{"at_us": 900_000_000, "device_id": rig.PIXEL,
                                                    "kind": "HELPER_LOST"}],
                      "request_results": [{"model_id": rig.QWEN, "output_tokens": 300, "trace_arrival_us": 0,
                                           "actual_latency_us": 120_000_000}]}
            config = MeasuredPlacementConfig(rig_profile="4060ti-op15-pixel-v1")
            record = run_shadow(config, run, result, {value: key for key, value in paths.items()})
            self.assertEqual(record["event_errors"], [])
            self.assertEqual(record["event_counts"]["SERVER_LAUNCH"], len(servers))
            self.assertEqual(record["event_counts"]["DEVICE_LEFT"], 1)
            self.assertIn("MASK_OUT_NOW", record["controller"]["decision_counts"])
            stored = json.loads((run / "LAYER_PLACEMENT.json").read_text())
            self.assertEqual(stored["configuration"]["mode"], "shadow")
            profile = LayerPlacementProfile.from_json(json.loads((run / "LAYER_PLACEMENT_PROFILE.json").read_text()))
            self.assertGreater(profile.revision, PROFILE.revision)
            self.assertEqual(record["initial_report"]["ideal"]["models"][rig.GEMMA]["owners"][rig.OP15]["layers"],
                             "0-26")


if __name__ == "__main__":
    unittest.main()
