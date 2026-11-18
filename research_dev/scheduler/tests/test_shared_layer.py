"""Shared FFN layers: the fit of a share sweep, bracketed savings, the static share plan, the profile fold."""
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
from research_dev.scheduler._internal.shared_layer import (  # noqa: E402
    PhoneCapacity,
    SharedLayerFit,
    StaticSharePlan,
    bracketed_deltas,
    fit_shared_layer,
    fold_fit_into_profile,
    plan_static_share,
    pool_deltas,
    share_points,
)

N_FF = 25600
N_EMBD = 5120
# One measured sweep of a 64-layer f16 model (18 CPU layers) with an NPU helper over direct USB:
# columns, wall ms of the shared layer, host share ms, phone round trip ms, phone compute ms; 504 calls each.
MEASURED_SWEEP = (
    (3200, 8.631628, 8.626138, 3.164392, 1.769986),
    (6400, 7.430659, 7.429191, 5.028763, 3.420345),
    (9600, 6.697511, 6.255583, 6.553033, 5.095583),
    (12800, 8.205786, 5.047957, 8.091717, 6.746544),
    (16000, 9.852715, 3.879394, 9.742075, 8.390641),
    (19200, 11.519023, 2.730358, 11.403718, 10.029071),
    (22400, 13.305184, 1.464227, 13.166126, 11.682353),
    (25600, 14.741576, 0.006994, 14.641835, 13.35277),
)


def shape(columns, layer_ms, host_ms, rpc_ms, compute_ms, *, tokens=1, calls=504):
    return {"tokens": tokens, "columns": columns, "calls": calls, "overlap_mean_ms": layer_ms,
            "host_mean_ms": host_ms, "rpc_mean_ms": rpc_ms, "compute_mean_ms": compute_ms}


def model_rows(t_host, c_host, t_link, t_phone, shares, n_ff=N_FF):
    rows = []
    for share in shares:
        host = c_host + (1 - share) * t_host if share < 1 else 0.0
        rpc = t_link + share * t_phone
        rows.append(shape(int(share * n_ff), max(host, rpc), host, rpc, share * t_phone))
    return rows


def measured_fit():
    return fit_shared_layer([shape(*row) for row in MEASURED_SWEEP], n_ff=N_FF)


class SharedLayerFitTests(unittest.TestCase):
    def test_fit_recovers_the_model_that_made_the_rows(self):
        fit = fit_shared_layer(model_rows(10.0, 0.4, 1.5, 12.0, (0.125, 0.25, 0.375, 0.5, 1.0)), n_ff=N_FF)
        for name, value in (("t_host_ms", 10.0), ("c_host_ms", 0.4), ("t_link_ms", 1.5), ("t_phone_ms", 12.0)):
            self.assertAlmostEqual(getattr(fit, name), value, places=9)
        self.assertAlmostEqual(fit.f_star, (10.0 + 0.4 - 1.5) / 22.0, places=9)
        self.assertAlmostEqual(fit.layer_ms(fit.f_star), fit.host_ms(fit.f_star), places=9)
        self.assertAlmostEqual(fit.layer_ms(fit.f_star), fit.phone_ms(fit.f_star), places=9)
        self.assertEqual(fit.layer_ms(0.0), 10.0)
        self.assertAlmostEqual(fit.gain_ms(1.0), 10.0 - 13.5, places=9)

    def test_measured_sweep_gives_the_recorded_fit(self):
        fit = measured_fit()
        self.assertAlmostEqual(fit.t_host_ms, 9.502739, places=5)
        self.assertAlmostEqual(fit.c_host_ms, 0.310466, places=5)
        self.assertAlmostEqual(fit.t_link_ms, 1.622279, places=5)
        self.assertAlmostEqual(fit.t_phone_ms, 13.069650, places=5)
        self.assertAlmostEqual(fit.f_star, 0.362874, places=5)
        self.assertAlmostEqual(fit.f_plain, 0.349119, places=5)
        self.assertAlmostEqual(fit.layer_ms(fit.f_star), 6.364911, places=5)
        # the model follows the measured layer within 0.25 ms at every share
        for point in fit.points:
            self.assertLess(abs(fit.layer_ms(point.share) - point.layer_ms), 0.25)

    def test_fit_round_trips_through_json(self):
        fit = measured_fit()
        again = SharedLayerFit.from_json(fit.to_json())
        self.assertEqual(again, fit)
        self.assertEqual(again.to_json(), fit.to_json())

    def test_rows_of_other_token_counts_are_ignored_and_duplicates_merge_by_calls(self):
        rows = model_rows(10.0, 0.4, 1.5, 12.0, (0.25, 0.5, 0.75))
        rows.append(shape(6400, 99.0, 99.0, 99.0, 99.0, tokens=2))
        rows.append(shape(6400, 9.0, 9.9, 6.5, 5.0, calls=0))
        twice = rows + [shape(12800, 8.5, 5.4, 8.5, 6.0, calls=1512)]
        points = share_points(twice, n_ff=N_FF)
        self.assertEqual([point.columns for point in points], [6400, 12800, 19200])
        self.assertEqual(points[1].calls, 2016)
        self.assertAlmostEqual(points[1].rpc_ms, (7.5 * 504 + 8.5 * 1512) / 2016, places=9)
        self.assertEqual(len(share_points(twice, n_ff=N_FF, rows=2)), 1)

    def test_thin_or_degenerate_sweeps_fail_closed(self):
        with self.assertRaisesRegex(LayerPlacementError, "at least 3 shares"):
            fit_shared_layer(model_rows(10.0, 0.4, 1.5, 12.0, (0.25, 0.5)), n_ff=N_FF)
        with self.assertRaisesRegex(LayerPlacementError, "at least 3 shares"):
            # three shares, but the full width leaves the host no work: two host points only
            fit_shared_layer(model_rows(10.0, 0.4, 1.5, 12.0, (0.25, 0.5, 1.0)), n_ff=N_FF)
        with self.assertRaisesRegex(LayerPlacementError, "outside"):
            fit_shared_layer([shape(N_FF + 32, 1.0, 1.0, 1.0, 1.0)], n_ff=N_FF)
        with self.assertRaisesRegex(LayerPlacementError, "lacks host_mean_ms"):
            fit_shared_layer([{"tokens": 1, "columns": 3200, "calls": 5, "rpc_mean_ms": 1.0,
                               "compute_mean_ms": 1.0}], n_ff=N_FF)
        falling = [shape(3200 * k, 5.0, 9.0 - k, 9.0 - k, 1.0) for k in (1, 2, 3, 4)]
        with self.assertRaisesRegex(LayerPlacementError, "growing with the share"):
            fit_shared_layer(falling, n_ff=N_FF)
        with self.assertRaisesRegex(LayerPlacementError, "negative link time"):
            fit_shared_layer(model_rows(10.0, 0.4, -0.5, 12.0, (0.25, 0.5, 0.75)), n_ff=N_FF)


def leg(columns, ms_token, host_j, *, phone_w=1.0, cpu_w=200.0, gpu_w=100.0, n_ff=N_FF):
    return {"columns": columns, "share": columns / n_ff, "ms_token": ms_token, "host_j_token": host_j,
            "total_j_token": host_j + (phone_w or 0.0) * ms_token * 1e-3, "cpu_w": cpu_w, "gpu_w": gpu_w,
            "phone_w": phone_w}


class BracketedDeltaTests(unittest.TestCase):
    def test_shared_leg_is_compared_with_both_neighbouring_host_legs(self):
        legs = [leg(0, 300.0, 90.0), leg(9600, 240.0, 74.0, phone_w=2.0), leg(0, 320.0, 96.0)]
        (row,) = bracketed_deltas(legs)
        self.assertEqual(row["brackets"], 2)
        self.assertAlmostEqual(row["base_ms"], 310.0)
        self.assertAlmostEqual(row["d_time"], 240.0 / 310.0 - 1.0)
        self.assertAlmostEqual(row["base_host_j"], 93.0)
        self.assertAlmostEqual(row["host_dj"], 19.0)
        self.assertAlmostEqual(row["phone_j"], 0.48)
        self.assertAlmostEqual(row["d_energy"], (74.0 + 0.48) / (93.0 + 0.31) - 1.0)
        self.assertTrue(row["phone_metered"])

    def test_host_drift_between_brackets_does_not_enter_the_saving(self):
        # the host slows by 10 % over the run; a shared leg that changes nothing must show no change
        legs = [leg(0, 300.0, 90.0), leg(3200, 315.0, 94.5), leg(0, 330.0, 99.0)]
        (row,) = bracketed_deltas(legs)
        self.assertAlmostEqual(row["d_time"], 0.0)

    def test_single_bracket_and_missing_bracket(self):
        (row,) = bracketed_deltas([leg(0, 300.0, 90.0), leg(9600, 240.0, 74.0)])
        self.assertEqual(row["brackets"], 1)
        with self.assertRaisesRegex(LayerPlacementError, "no host-only leg"):
            bracketed_deltas([leg(9600, 240.0, 74.0)])

    def test_unmetered_phone_adds_no_energy_and_is_flagged(self):
        legs = [leg(0, 300.0, 90.0, phone_w=None), leg(9600, 240.0, 74.0, phone_w=None)]
        (row,) = bracketed_deltas(legs)
        self.assertEqual(row["phone_j"], 0.0)
        self.assertFalse(row["phone_metered"])
        self.assertFalse(pool_deltas([row])["0.375"]["phone_metered"])

    def test_pooled_rows_and_marginal_host_power(self):
        legs = [leg(0, 300.0, 90.0), leg(9600, 240.0, 72.0), leg(0, 300.0, 90.0), leg(9600, 250.0, 75.0),
                leg(0, 300.0, 90.0), leg(3200, 290.0, 88.0), leg(0, 300.0, 90.0)]
        pooled = pool_deltas(bracketed_deltas(legs))
        self.assertEqual(sorted(pooled), ["0.125", "0.375"])
        best = pooled["0.375"]
        self.assertEqual(best["n_legs"], 2)
        self.assertAlmostEqual(best["time_saving_mean"], (20.0 + 100.0 / 6) / 2)
        self.assertAlmostEqual(best["time_saving_min"], 100.0 / 6)
        self.assertAlmostEqual(best["shared"]["ms"], 245.0)
        self.assertAlmostEqual(best["alone"]["total_j"], 90.0)
        # host joules saved per second saved: 18 J / 0.060 s and 15 J / 0.050 s
        self.assertAlmostEqual(best["marginal_host_w"], 300.0)
        self.assertAlmostEqual(best["energy_saving_mean"],
                               ((1 - 72.24 / 90.0) + (1 - 75.25 / 90.0)) * 50.0)


class StaticSharePlanTests(unittest.TestCase):
    def plan(self, capacity, **overrides):
        arguments = {"cpu_layers": range(18), "n_embd": N_EMBD, "column_quantum": 32, "weight_type": "F16",
                     "capacity": capacity, **overrides}
        return plan_static_share(measured_fit(), **arguments)

    def test_roomy_phone_gets_the_optimum_on_every_layer(self):
        plan = self.plan(PhoneCapacity(8_000_000_000, sessions=3, session_limit_bytes=3_300_000_000))
        self.assertEqual((plan.columns, len(plan.layers), plan.binding_constraint), (9600, 18, "optimum"))
        self.assertEqual(plan.sessions, (tuple(range(9)), tuple(range(9, 18))))
        self.assertEqual(plan.layer_bytes, ffn_layer_bytes(N_EMBD, 9600, "F16"))
        self.assertEqual(plan.resident_bytes, 18 * plan.layer_bytes)
        # 18 layers x (9.50 - 6.52) ms; the run this sweep belongs to measured 53.1 ms per token
        self.assertAlmostEqual(plan.predicted_token_ms_saved, 53.6, places=1)
        self.assertTrue(plan.beneficial)

    def test_one_session_keeps_a_layer_subset_at_a_useful_share(self):
        plan = self.plan(PhoneCapacity(3_300_000_000, sessions=1, session_limit_bytes=3_300_000_000))
        self.assertEqual(plan.binding_constraint, "memory")
        self.assertLessEqual(plan.resident_bytes, 3_300_000_000)
        self.assertEqual(len(plan.sessions), 1)
        self.assertEqual(plan.layers, tuple(range(len(plan.layers))))
        # a subset at a larger share beats a quarter share on more layers here
        quarter = next(row for row in plan.alternatives if row["columns"] == 6400)
        self.assertGreater(plan.predicted_token_ms_saved, quarter["token_ms_saved"])
        self.assertGreater(plan.columns, 6400)

    def test_session_limit_binds_when_memory_does_not(self):
        plan = self.plan(PhoneCapacity(64_000_000_000, sessions=1, session_limit_bytes=3_300_000_000))
        self.assertEqual(plan.binding_constraint, "session")
        self.assertLess(len(plan.layers), 18)

    def test_slow_link_is_not_beneficial(self):
        slow = SharedLayerFit(1, N_FF, 5.0, 0.3, 6.0, 13.0)
        plan = plan_static_share(slow, cpu_layers=range(18), n_embd=N_EMBD, column_quantum=32, weight_type="F16",
                                 capacity=PhoneCapacity(8_000_000_000))
        self.assertEqual((plan.binding_constraint, plan.columns, plan.layers, plan.sessions),
                         ("not-beneficial", 0, (), ()))
        self.assertFalse(plan.beneficial)
        self.assertEqual(plan.predicted_layer_ms, 5.0)
        self.assertTrue(plan.alternatives)

    def test_phone_without_room_for_one_layer_is_not_beneficial(self):
        plan = self.plan(PhoneCapacity(10_000_000))
        self.assertEqual(plan.binding_constraint, "not-beneficial")

    def test_candidates_respect_stored_columns_and_the_quantum(self):
        plan = self.plan(PhoneCapacity(8_000_000_000), stored_columns=6400)
        self.assertEqual(plan.columns, 6400)
        odd = self.plan(PhoneCapacity(8_000_000_000), column_quantum=3000)
        self.assertEqual(odd.columns % 3000, 0)
        with self.assertRaisesRegex(LayerPlacementError, "stored columns"):
            self.plan(PhoneCapacity(8_000_000_000), stored_columns=N_FF + 1)
        with self.assertRaisesRegex(LayerPlacementError, "no column share"):
            self.plan(PhoneCapacity(8_000_000_000), stored_columns=800)

    def test_plan_round_trips_through_json(self):
        plan = self.plan(PhoneCapacity(4_800_000_000, sessions=2, session_limit_bytes=3_300_000_000))
        again = StaticSharePlan.from_json(plan.to_json())
        self.assertEqual(again.to_json(), plan.to_json())


class ProfileFoldTests(unittest.TestCase):
    def test_planner_prices_the_layer_as_the_fit_does(self):
        fit = measured_fit()
        layer_bytes = ffn_layer_bytes(N_EMBD, N_FF, "F16")
        profile = LayerPlacementProfile(DesktopPower(100.0, 100.0))
        fold_fit_into_profile(profile, fit, model_id="model-a", device_id="helper-a", shard_format="f16",
                              transport="link-a", layer_bytes=layer_bytes, evidence="sweep-a",
                              host_ffn_w=300.0, phone_active_w=2.5, phone_idle_w=0.5)
        self.assertEqual(profile.desktop_power.ffn_w, 300.0)
        self.assertEqual(profile.desktop_power.wait_w, 100.0)
        cost = profile.call_cost("helper-a", "model-a", "f16", "link-a", 1, layer_bytes)
        self.assertAlmostEqual(cost.rpc_ms.value, fit.t_link_ms + fit.t_phone_ms)
        self.assertAlmostEqual(cost.compute_ms.value, fit.t_phone_ms)
        self.assertEqual(cost.rpc_ms.provenance, "derived")
        priors = profile.device_priors["helper-a"]
        self.assertAlmostEqual(priors.bytes_per_s["f16"], layer_bytes / (fit.t_phone_ms / 1000.0))
        self.assertAlmostEqual(priors.overhead("link-a", 1), fit.t_link_ms)
        self.assertEqual((priors.active_marginal_w, priors.idle_w), (2.0, 0.5))
        model = ModelLayers("model-a", tuple(range(4)), MappingProxyType({"f16": layer_bytes}), n_embd=N_EMBD,
                            n_ff=N_FF, column_quantum=3200)
        device = HelperDevice("helper-a", MappingProxyType({"model-a": "f16"}), 8 * layer_bytes, "link-a")
        problem = PlacementProblem((model,), (device,), profile,
                                   column_fractions=MappingProxyType({"model-a": (0.375,)}))
        placed = solve_placement(problem).models["model-a"]
        self.assertEqual(placed.column_fraction, 0.375)
        self.assertEqual(set(placed.owners.values()), {"helper-a"})
        # the planner's layer is the fit's without the fixed cost of sharing
        self.assertAlmostEqual(placed.chain_ms_by_rows[1], 4 * max(0.625 * fit.t_host_ms, fit.phone_ms(0.375)))
        self.assertAlmostEqual(placed.cpu_only_chain_ms_by_rows[1], 4 * fit.t_host_ms)
        # the profile survives its own JSON round trip with the folded rows
        again = LayerPlacementProfile.from_json(profile.to_json())
        self.assertEqual(again.to_json(), profile.to_json())

    def test_second_fold_keeps_other_formats_and_transports(self):
        fit = measured_fit()
        profile = LayerPlacementProfile(DesktopPower(100.0, 100.0))
        fold_fit_into_profile(profile, fit, model_id="model-a", device_id="helper-a", shard_format="f16",
                              transport="link-a", layer_bytes=1_000_000, phone_active_w=2.5, phone_idle_w=0.5)
        fold_fit_into_profile(profile, fit, model_id="model-b", device_id="helper-a", shard_format="q4_0",
                              transport="link-b", layer_bytes=500_000)
        priors = profile.device_priors["helper-a"]
        self.assertEqual(sorted(priors.bytes_per_s), ["f16", "q4_0"])
        self.assertEqual(sorted(priors.overhead_ms), ["link-a", "link-b"])
        self.assertEqual((priors.active_marginal_w, priors.idle_w), (2.0, 0.5))
        with self.assertRaisesRegex(LayerPlacementError, "layer bytes"):
            fold_fit_into_profile(profile, fit, model_id="model-a", device_id="helper-a", shard_format="f16",
                                  transport="link-a", layer_bytes=0)


if __name__ == "__main__":
    unittest.main()
