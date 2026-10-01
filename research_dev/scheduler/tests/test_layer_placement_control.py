"""WS11 rebalancing: events, mandatory mask-outs, runtime vs next-launch vs restart, hysteresis, no thrash."""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler._internal.layer_placement import (  # noqa: E402
    DesktopPower,
    DeviceCostPriors,
    HelperDevice,
    LayerPlacementError,
    LayerPlacementProfile,
    ModelLayers,
)
from research_dev.scheduler._internal.layer_placement_control import (  # noqa: E402
    LayerPlacementController,
    MeasuredPlacementConfig,
    PlacementEvent,
    RebalancePolicy,
    measured_placement_from_policy_json,
)
from research_dev.scheduler.configuration import campaign as campaign_config  # noqa: E402


ALL = 0b1111


def priors(rate, overhead, transport="usb"):
    return DeviceCostPriors(
        bytes_per_s=MappingProxyType({"f16": rate}),
        overhead_ms=MappingProxyType({transport: MappingProxyType({1: overhead})}),
        rows_factor=MappingProxyType({1: 1.0}), active_marginal_w=10.0, idle_w=0.5)


def store() -> LayerPlacementProfile:
    # fast/cheap "a" (10 ms), slower "b" (14 ms), CPU 20 ms per layer at 150 W
    return LayerPlacementProfile(desktop_power=DesktopPower(150.0, 50.0, 25.0, "measured"),
                                 desktop_bytes_per_s=25e9,
                                 device_priors={"a": priors(55e9, 0.9), "b": priors(40e9, 1.5)})


def model(name="m", count=4) -> ModelLayers:
    return ModelLayers(name, tuple(range(count)), MappingProxyType({"f16": 500_000_000}))


def device(name, *, capacity=10**12, stored=ALL, qualified=ALL, model_id="m"):
    return HelperDevice(name, MappingProxyType({model_id: "f16"}), capacity, "usb",
                        stored_layers=MappingProxyType({model_id: stored}),
                        qualified_layers=MappingProxyType({model_id: qualified}))


def controller(devices, *, initial=None, policy=None, models=None):
    return LayerPlacementController.create(
        models or [model()], devices, store(), initial=initial,
        policy=policy or RebalancePolicy(min_dwell_s=0.0, restart_j=MappingProxyType({"m": 2000.0})))


def actions(decisions):
    return [row.action for row in decisions]


class ControllerTests(unittest.TestCase):
    def test_start_adopts_the_measured_plan_for_the_next_launch(self):
        c = controller([device("a", capacity=1_000_000_000), device("b")])
        decisions = c.handle(PlacementEvent("START", 0.0))
        self.assertEqual(actions(decisions), ["ADOPT_NEXT_LAUNCH"])
        self.assertEqual(c.launch_masks("m"), {"a": 0b0011, "b": 0b1100})

    def test_loss_while_running_masks_out_then_relaunches_when_it_pays(self):
        c = controller([device("a"), device("b")], initial={"m": {0: "a", 1: "a", 2: "a", 3: "a"}})
        c.handle(PlacementEvent("SERVER_LAUNCH", 1.0, model_id="m"))
        self.assertEqual(c.runtime_layer_mask("m"), ALL)
        decisions = c.handle(PlacementEvent("DEVICE_LEFT", 2.0, device_id="a"))
        self.assertEqual(actions(decisions)[0], "MASK_OUT_NOW")
        # b could take all four layers, but llama-server binds owners at launch: the gain over the horizon
        # (900 tokens x ~4.9 J) pays the 2 kJ relaunch
        self.assertEqual(actions(decisions)[1], "RESTART_NOW")
        self.assertEqual(c.launch_masks("m"), {"b": ALL})
        self.assertEqual(c.runtime_layer_mask("m"), ALL)

    def test_loss_without_a_paying_relaunch_defers_to_the_next_launch(self):
        policy = RebalancePolicy(min_dwell_s=0.0, restart_j=MappingProxyType({"m": 10**7}))
        c = controller([device("a"), device("b")], initial={"m": dict.fromkeys(range(4), "a")}, policy=policy)
        c.handle(PlacementEvent("SERVER_LAUNCH", 1.0, model_id="m"))
        decisions = c.handle(PlacementEvent("DEVICE_LEFT", 2.0, device_id="a"))
        self.assertEqual(actions(decisions), ["MASK_OUT_NOW", "ADOPT_NEXT_LAUNCH"])
        self.assertEqual(c.runtime_layer_mask("m"), 0)            # host computes until the relaunch
        c.handle(PlacementEvent("SERVER_EXIT", 3.0, model_id="m"))
        c.handle(PlacementEvent("SERVER_LAUNCH", 4.0, model_id="m"))
        self.assertEqual(c.runtime_layer_mask("m"), ALL)
        self.assertEqual(c.launch_masks("m"), {"b": ALL})

    def test_readmission_restores_inside_the_launched_ownership_at_runtime(self):
        c = controller([device("a")], initial={"m": dict.fromkeys(range(4), "a")})
        c.handle(PlacementEvent("SERVER_LAUNCH", 1.0, model_id="m"))
        c.handle(PlacementEvent("DEVICE_QUARANTINED", 2.0, device_id="a"))
        self.assertEqual(c.runtime_layer_mask("m"), 0)
        decisions = c.handle(PlacementEvent("DEVICE_READMITTED", 3.0, device_id="a"))
        self.assertEqual(actions(decisions), ["ADOPT_RUNTIME"])
        self.assertEqual(c.runtime_layer_mask("m"), ALL)

    def test_thermal_exclusion_and_clearance(self):
        c = controller([device("a"), device("b")], initial={"m": dict.fromkeys(range(4), "a")})
        c.handle(PlacementEvent("THERMAL_EXCLUDED", 1.0, device_id="a"))
        self.assertEqual(c.launch_masks("m"), {"b": ALL})
        c.handle(PlacementEvent("THERMAL_CLEARED", 200.0, device_id="a"))
        self.assertEqual(c.launch_masks("m"), {"a": ALL})

    def test_hysteresis_min_dwell_and_margin(self):
        policy = RebalancePolicy(min_dwell_s=300.0, restart_j=MappingProxyType({"m": 2000.0}))
        c = controller([device("a"), device("b")], initial={"m": dict.fromkeys(range(4), "b")}, policy=policy)
        c.handle(PlacementEvent("START", 0.0))
        self.assertEqual(c.launch_masks("m"), {"a": ALL})
        c.handle(PlacementEvent("DEVICE_LEFT", 10.0, device_id="a"))     # mandatory: ignores the dwell
        self.assertEqual(c.launch_masks("m"), {"b": ALL})
        decisions = c.handle(PlacementEvent("DEVICE_JOINED", 20.0, device_id="a"))   # optional: waits
        self.assertEqual(actions(decisions), ["HOLD"])
        self.assertEqual(decisions[0].reason, "MIN_DWELL")
        c.handle(PlacementEvent("PERIODIC", 400.0))
        self.assertEqual(c.launch_masks("m"), {"a": ALL})
        # a running server: the relaunch is not paid by a short horizon, the next launch is free
        short = RebalancePolicy(min_dwell_s=0.0, horizon_s=1.0, default_tokens_per_s=1.0,
                                restart_j=MappingProxyType({"m": 2000.0}))
        c = controller([device("a"), device("b")], initial={"m": dict.fromkeys(range(4), "b")}, policy=short)
        c.handle(PlacementEvent("SERVER_LAUNCH", 0.0, model_id="m"))
        decisions = c.handle(PlacementEvent("PERIODIC", 1.0))
        self.assertEqual([(row.action, row.reason) for row in decisions],
                         [("ADOPT_NEXT_LAUNCH", "RELAUNCH_NOT_PAID_DEFERRED")])
        self.assertEqual(c.runtime_layer_mask("m"), ALL)      # still served by b until the relaunch
        # a gain below the minimum per token is not worth any change
        picky = RebalancePolicy(min_dwell_s=0.0, min_gain_j_per_token=100.0, restart_j=MappingProxyType({"m": 1.0}))
        c = controller([device("a"), device("b")], initial={"m": dict.fromkeys(range(4), "b")}, policy=picky)
        decisions = c.handle(PlacementEvent("PERIODIC", 1.0))
        self.assertEqual([(row.action, row.reason) for row in decisions], [("HOLD", "GAIN_BELOW_MARGIN")])

    def test_drift_replans_only_beyond_the_threshold_and_does_not_thrash(self):
        c = controller([device("a", capacity=1_000_000_000), device("b")])
        c.handle(PlacementEvent("START", 0.0))
        before = dict(c.launch_masks("m"))
        changes = 0
        for step in range(40):     # +-5 % noise on both helpers' measured calls: never above the 15 % drift
            factor = 1.05 if step % 2 else 0.95
            rows = [{"device_id": name, "model_id": "m", "shard_format": "f16", "transport": "usb", "rows": 1,
                     "calls": 500, "rpc_ms": base * factor, "compute_ms": (base - 1.0) * factor}
                    for name, base in (("a", 10.0), ("b", 14.0))]
            decisions = c.handle(PlacementEvent("SHAPES_OBSERVED", 10.0 * (step + 1), model_id="m",
                                                details={"rows": rows}))
            changes += sum(row.action not in ("HOLD", "KEEP") for row in decisions)
        self.assertEqual(c.launch_masks("m"), before)
        self.assertLessEqual(changes, 1)
        # a real change (b becomes much faster) is acted on
        rows = [{"device_id": "b", "model_id": "m", "shard_format": "f16", "transport": "usb", "rows": 1,
                 "calls": 50_000, "rpc_ms": 4.0, "compute_ms": 3.5}]
        c.handle(PlacementEvent("SHAPES_OBSERVED", 1000.0, model_id="m", details={"rows": rows}))
        self.assertEqual(c.launch_masks("m").get("b"), ALL)

    def test_new_device_joins_with_its_spec_and_wins_layers(self):
        c = controller([device("b")], initial={"m": dict.fromkeys(range(4), "b")})
        c.profile.device_priors["new"] = priors(80e9, 0.5)
        decisions = c.handle(PlacementEvent("DEVICE_JOINED", 5.0, device_id="new",
                                            details={"device": device("new")}))
        self.assertIn("ADOPT_NEXT_LAUNCH", actions(decisions))
        self.assertEqual(c.launch_masks("m"), {"new": ALL})
        with self.assertRaises(LayerPlacementError):
            c.handle(PlacementEvent("DEVICE_JOINED", 6.0, device_id="ghost"))

    def test_unqualified_or_unstored_layers_never_execute(self):
        # "a" is best everywhere but holds and qualifies only layers 0-1
        c = controller([device("a", stored=0b0011, qualified=0b0011), device("b")])
        decisions = c.handle(PlacementEvent("START", 0.0))
        self.assertIn("BLOCKED_QUALIFICATION", actions(decisions))
        self.assertEqual(c.launch_masks("m")["a"], 0b0011)
        # shards pushed and qualified -> the ideal plan becomes executable
        c.handle(PlacementEvent("SHARDS_STORED", 100.0, device_id="a", model_id="m",
                                details={"layer_mask": 0b1100, "qualified_mask": 0b1100}))
        self.assertEqual(c.launch_masks("m"), {"a": ALL})
        # stored but unqualified: the push is recommended only when the evidence exists
        c = controller([device("a", stored=0b0011, qualified=ALL), device("b")])
        decisions = c.handle(PlacementEvent("START", 0.0))
        self.assertIn("PROVISION", actions(decisions))

    def test_model_added_and_removed(self):
        shared = HelperDevice("a", MappingProxyType({"m": "f16", "n": "f16"}), 2_000_000_000, "usb",
                              stored_layers=MappingProxyType({"m": ALL, "n": ALL}),
                              qualified_layers=MappingProxyType({"m": ALL, "n": ALL}))
        c = controller([shared], initial={"m": dict.fromkeys(range(4), "a")})
        c.handle(PlacementEvent("MODEL_ADDED", 1.0, details={"model": model("n")}))
        total = sum(bin(c.launch_masks(name).get("a", 0)).count("1") for name in ("m", "n"))
        self.assertEqual(total, 4)                  # 2 GB co-resident capacity shared by both models
        c.handle(PlacementEvent("MODEL_REMOVED", 2.0, model_id="n"))
        self.assertEqual(c.launch_masks("m"), {"a": ALL})
        self.assertNotIn("n", c.target)

    def test_envelope_grows_from_sustained_measurements(self):
        policy = RebalancePolicy(min_dwell_s=0.0, busy_growth_ppm=250_000, sustained_calls=100,
                                 restart_j=MappingProxyType({"m": 2000.0}))
        spec = replace(device("a"), busy_envelope_ms=10.0)
        c = controller([spec], policy=policy)
        c.handle(PlacementEvent("START", 0.0))
        self.assertEqual(c.launch_masks("m"), {"a": 0b0001})         # 12.5 ms cap: one 10 ms call
        c.handle(PlacementEvent("SERVER_LAUNCH", 1.0, model_id="m"))
        rows = [{"device_id": "a", "model_id": "m", "shard_format": "f16", "transport": "usb", "rows": 1,
                 "calls": 5000, "rpc_ms": 10.0, "compute_ms": 9.1}]
        c.handle(PlacementEvent("SHAPES_OBSERVED", 50.0, model_id="m", details={"rows": rows}))
        c.handle(PlacementEvent("SERVER_EXIT", 51.0, model_id="m"))
        self.assertEqual(c.envelope_ms["a"], 10.0)
        c.envelope_ms["a"] = 30.0                                  # e.g. after a sustained 3-layer run
        c.handle(PlacementEvent("PERIODIC", 100.0))
        self.assertEqual(bin(c.launch_masks("m")["a"]).count("1"), 3)

    def test_summary_is_json(self):
        c = controller([device("a")])
        c.handle(PlacementEvent("START", 0.0))
        record = c.to_json()
        self.assertEqual(record["schema"], "research-scheduler-layer-placement-control-v1")
        self.assertIn("ADOPT_NEXT_LAUNCH", record["decision_counts"])
        with self.assertRaises(LayerPlacementError):
            PlacementEvent("BOGUS", 0.0)


class ConfigTests(unittest.TestCase):
    def test_measured_placement_key_is_opt_in_and_validated(self):
        legacy = {"continuous_join": True, "max_barrier_extension_s": 120, "model_affinity": True,
                  "residency_hysteresis_s": 20, "work_conserving_admission": True}
        self.assertEqual(dict(campaign_config._dispatch_policy(legacy)), legacy)
        enabled = {**legacy, "measured_placement": {"mode": "shadow", "rig_profile": "4060ti-op15-pixel-v1"}}
        checked = campaign_config._dispatch_policy(enabled)
        self.assertEqual(checked["measured_placement"], {"mode": "shadow", "rig_profile": "4060ti-op15-pixel-v1"})
        rest, config = measured_placement_from_policy_json(dict(checked))
        self.assertEqual(rest, legacy)
        self.assertEqual(config.policy().busy_growth_ppm, 250_000)
        for bad in ({"mode": "active", "rig_profile": "4060ti-op15-pixel-v1"}, {"mode": "shadow"},
                    {"mode": "shadow", "rig_profile": "4060ti-op15-pixel-v1", "margin_ppm": -1},
                    {"mode": "shadow", "profile_path": "relative.json", "inventory_path": "/x.json"},
                    {"mode": "shadow", "rig_profile": "4060ti-op15-pixel-v1", "surprise": 1}):
            with self.assertRaises(ValueError, msg=str(bad)):
                campaign_config._dispatch_policy({**legacy, "measured_placement": bad})
        self.assertEqual(MeasuredPlacementConfig.from_json(config.to_json()), config)

    def test_runner_strips_the_key_before_the_runtime_policy(self):
        import json
        from research_dev.scheduler.campaigns.burstgpt import runner
        text = json.dumps({"work_conserving_admission": True,
                           "measured_placement": {"mode": "shadow", "rig_profile": "4060ti-op15-pixel-v1"}})
        policy = runner._dispatch_policy_from_json(text)
        self.assertTrue(policy.work_conserving_admission)
        self.assertIsNotNone(runner._measured_placement_from_json(text))
        only = json.dumps({"measured_placement": {"mode": "shadow", "rig_profile": "4060ti-op15-pixel-v1"}})
        self.assertIsNone(runner._dispatch_policy_from_json(only))
        self.assertIsNone(runner._measured_placement_from_json(json.dumps({"work_conserving_admission": True})))


if __name__ == "__main__":
    unittest.main()
