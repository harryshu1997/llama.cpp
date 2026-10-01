"""WS11 new resources / new models: capability probe -> profile, fail-closed model onboarding."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import MappingProxyType

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler._internal.layer_placement import (  # noqa: E402
    DesktopPower,
    HelperDevice,
    LayerPlacementError,
    LayerPlacementProfile,
    ModelLayers,
    PlacementProblem,
    ffn_layer_bytes,
    solve_placement,
)
from research_dev.scheduler._internal.layer_placement_onboarding import (  # noqa: E402
    DeviceCapability,
    helper_device_for,
    onboard_model,
    probe_plan,
    profile_from_probe,
)


def tensor(weight_type, shape):
    return {"type": weight_type, "shape": list(shape)}


def metadata(architecture="qwen3", layers=8, n_embd=512, n_ff=1536, weight="F16", moe=(), down=None):
    rows = []
    for index in range(layers):
        rows.append({"index": index, "moe": index in moe,
                     "gate": tensor(weight, (n_embd, n_ff)), "up": tensor(weight, (n_embd, n_ff)),
                     "down": tensor((down or {}).get(index, weight), (n_ff, n_embd))})
    return {"model_id": "toy", "architecture": architecture, "block_count": layers, "n_embd": n_embd,
            "n_ff": n_ff, "layers": rows}


HTP = DeviceCapability("op15-phone", "htp", ("f16",), 9_625_939_968, "functionfs-usb", 3, 3_208_646_656, "per-model")
PIXEL = DeviceCapability("pixel10pro-phone", "pixel-packed", ("q4k-packed", "q4_0-packed", "f16"), 12 * 1024 ** 3,
                         "adb-tcp")


class OnboardingTests(unittest.TestCase):
    def test_splittable_model_and_cpu_layers_from_ngl(self):
        verdict = onboard_model(metadata(layers=8), n_gpu_layers=5, devices=[HTP])
        self.assertTrue(verdict.placeable)
        self.assertEqual(verdict.cpu_layers, (0, 1, 2, 3))     # 8 + 1 - 5 = 4 CPU blocks
        self.assertEqual(verdict.model.layer_bytes["f16"], ffn_layer_bytes(512, 1536, "F16"))
        self.assertEqual(verdict.devices[0].shard_format, "f16")
        # the rig's numbers: Qwen -ngl 16 of 41 -> 25 CPU layers, Gemma -ngl 22 of 49 -> 27
        self.assertEqual(len(onboard_model(metadata(layers=40, n_embd=512, n_ff=1536), n_gpu_layers=16,
                                           devices=[HTP]).cpu_layers), 25)
        self.assertEqual(len(onboard_model(metadata("gemma4", layers=48), n_gpu_layers=22, devices=[HTP]).cpu_layers), 27)

    def test_unsupported_architecture_moe_shape_and_mask_limit_fail_closed(self):
        for meta, ngl, reason in (
                (metadata("qwen2"), 4, "ARCHITECTURE_UNSUPPORTED"),
                (metadata("gemma4", moe=(1,)), 4, "MOE_LAYER"),
                (dict(metadata(), n_ff=1024), 4, "FFN_SHAPE"),
                (metadata(layers=80), 1, "LAYER_MASK_LIMIT"),
                (metadata(layers=8), 9, "NO_CPU_LAYERS")):
            verdict = onboard_model(meta, n_gpu_layers=ngl, devices=[HTP])
            self.assertFalse(verdict.placeable)
            self.assertTrue(any(row.startswith(reason) for row in verdict.reasons), (reason, verdict.reasons))

    def test_device_formats_and_origin_requirements(self):
        # host F16 = the dequantization of a Q4_K/Q6_K origin (the Pixel's packed path)
        origin = metadata(weight="Q4_K", n_ff=1536, down={1: "Q6_K"})
        verdict = onboard_model(metadata(), n_gpu_layers=5, devices=[PIXEL], origin_metadata=origin,
                                origin_dequantizes_exactly=True)
        row = verdict.devices[0]
        self.assertEqual(row.shard_format, "q4k-packed")
        self.assertEqual(verdict.model.layer_bytes["q4k-packed"], ffn_layer_bytes(512, 1536, "Q4_K"))
        self.assertEqual(verdict.model.layer_bytes_overrides["q4k-packed"][1],
                         ffn_layer_bytes(512, 1536, "Q4_K", "Q4_K", "Q6_K"))
        # without the exact-dequant proof the packed formats are refused; f16 is the fallback
        verdict = onboard_model(metadata(), n_gpu_layers=5, devices=[PIXEL], origin_metadata=origin)
        self.assertEqual(verdict.devices[0].shard_format, "f16")
        # a Q4_0 origin selects the Q4_0 packed path
        verdict = onboard_model(metadata(), n_gpu_layers=5, devices=[PIXEL], origin_metadata=metadata(weight="Q4_0"),
                                origin_dequantizes_exactly=True)
        self.assertEqual(verdict.devices[0].shard_format, "q4_0-packed")
        # a quantized host model on the HTP is approximate numerics: refused unless allowed
        q4 = metadata(weight="Q4_0")
        refused = onboard_model(q4, n_gpu_layers=5, devices=[DeviceCapability("op15-phone", "htp", ("q4_0",),
                                                                             9 * 10 ** 9, "usb")])
        self.assertFalse(refused.devices[0].ok)
        allowed = onboard_model(q4, n_gpu_layers=5, devices=[DeviceCapability("op15-phone", "htp", ("q4_0",),
                                                                             9 * 10 ** 9, "usb",
                                                                             allow_approximate=True)])
        self.assertTrue(allowed.devices[0].ok and allowed.devices[0].approximate_numerics)

    def test_memory_fit_per_session(self):
        big = metadata(layers=30, n_embd=5120, n_ff=17408)        # Qwen-sized f16 layers
        verdict = onboard_model(big, n_gpu_layers=1, devices=[HTP])
        self.assertEqual(verdict.devices[0].max_layers, 18)        # 6 per session x 3
        tiny = DeviceCapability("watch", "cpu-worker", ("f16",), 100_000_000, "ble")
        refused = onboard_model(big, n_gpu_layers=1, devices=[tiny])
        self.assertFalse(refused.devices[0].ok)
        self.assertIn("does not fit", refused.devices[0].reasons[0])
        with self.assertRaises(LayerPlacementError):
            helper_device_for(refused.devices[0], tiny, "toy")
        spec = helper_device_for(verdict.devices[0], HTP, "toy")
        self.assertEqual(spec.formats, {"toy": "f16"})
        base = HelperDevice("op15-phone", MappingProxyType({"other": "f16"}), 1, "usb", busy_envelope_ms=5.0)
        merged = helper_device_for(verdict.devices[0], HTP, "toy", base)
        self.assertEqual(dict(merged.formats), {"other": "f16", "toy": "f16"})
        self.assertEqual(merged.busy_envelope_ms, 5.0)


class ProbeTests(unittest.TestCase):
    def calls(self):
        rows = []
        for rows_count, compute_us, rpc_us in ((1, 6000, 6800), (2, 6600, 7600), (4, 9000, 10400)):
            rows.extend({"step": -1, "rows": rows_count, "compute_us": 99999, "rpc_us": 999999} for _ in range(4))
            rows.extend({"step": i, "rows": rows_count, "compute_us": compute_us, "rpc_us": rpc_us} for i in range(96))
        return rows

    def test_probe_registers_a_new_helper_that_the_planner_then_uses(self):
        profile = LayerPlacementProfile(desktop_power=DesktopPower(150.0, 50.0, 25.0, "measured"),
                                        desktop_bytes_per_s=25e9)
        plan = probe_plan("tablet", "f16", "usb-ncm", 300_000_000)
        self.assertEqual(plan.to_json()["total_calls"], 3 * (96 + 12))
        result = profile_from_probe(profile, plan, "m", self.calls(), meter_j_above_idle=17.28, idle_w=0.6,
                                    evidence="probe-1")
        self.assertAlmostEqual(result.by_rows[1]["compute_ms"], 6.0)
        self.assertAlmostEqual(result.priors.bytes_per_s["f16"], 300_000_000 / 0.006)
        self.assertAlmostEqual(result.priors.overhead_ms["usb-ncm"][4], 1.4)
        self.assertAlmostEqual(result.priors.rows_factor[4], 1.5)
        self.assertAlmostEqual(result.priors.active_marginal_w, 17.28 / ((6.0 + 6.6 + 9.0) * 96 / 1000))
        self.assertEqual(profile.calls[next(iter(profile.calls))].provenance, "measured")
        model = ModelLayers("m", (0, 1, 2), MappingProxyType({"f16": 300_000_000}))
        device = HelperDevice("tablet", MappingProxyType({"m": "f16"}), 10**12, "usb-ncm")
        placed = solve_placement(PlacementProblem((model,), (device,), profile))
        self.assertEqual(placed.helper_masks("m"), {"tablet": 0b111})
        self.assertEqual(placed.priors_used, ())

    def test_probe_rejects_thin_or_inconsistent_data(self):
        profile = LayerPlacementProfile(desktop_power=DesktopPower(150.0, 50.0, 25.0))
        plan = probe_plan("tablet", "f16", "usb", 300_000_000)
        with self.assertRaises(LayerPlacementError):
            profile_from_probe(profile, plan, "m", self.calls()[:50])
        bad = [dict(row, rpc_us=1) for row in self.calls()]
        with self.assertRaises(LayerPlacementError):
            profile_from_probe(profile, plan, "m", bad)
        with self.assertRaises(LayerPlacementError):
            probe_plan("tablet", "f16", "usb", 0)


if __name__ == "__main__":
    unittest.main()
