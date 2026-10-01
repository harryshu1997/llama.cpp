"""Paired-binary statistics of the quality noninferiority protocol against published and independent values."""
import math
import unittest

from research_dev.scheduler.campaigns.burstgpt.quality.stats import (
    PairedCounts, exact_mcnemar_p, newcombe_interval, noninferiority, noninferiority_power,
    noninferiority_sample_size, noninferiority_sample_size_normal, paired_bootstrap_interval,
    tango_interval, tango_restricted_loss, tango_z,
)

# Fagerland, Lydersen and Laake (2014, Stat Med 33:2850), Table V: Bentur et al. (2009), 21 children,
# AHR before (second) and after (first) transplantation: n11=1, n12=1, n21=7, n22=12, estimate -0.286.
BENTUR = PairedCounts(both=1, gain=1, loss=7, neither=12)


def brute_force_interval(counts, confidence_z, step=1e-5):
    """Independent inversion of the score test on a fine grid (no bisection, no monotonicity assumption)."""
    inside = [k * step for k in range(int(-1 / step) + 1, int(1 / step))
              if abs(tango_z(counts.n, counts.gain, counts.loss, k * step)) <= confidence_z]
    return min(inside), max(inside)


class QualityStatsTests(unittest.TestCase):
    def test_tango_interval_reproduces_the_published_example(self):
        self.assertAlmostEqual(BENTUR.difference, -0.286, places=3)
        low, high = tango_interval(BENTUR)
        self.assertEqual((round(low, 3), round(high, 3)), (-0.517, -0.026))

    def test_newcombe_interval_reproduces_the_published_example(self):
        low, high = newcombe_interval(BENTUR)
        self.assertEqual((round(low, 3), round(high, 3)), (-0.507, -0.026))

    def test_exact_mcnemar_reproduces_the_published_example(self):
        self.assertAlmostEqual(exact_mcnemar_p(BENTUR), 18 / 256)
        self.assertEqual(round(exact_mcnemar_p(BENTUR), 3), 0.070)
        self.assertEqual(exact_mcnemar_p(PairedCounts(both=5, gain=0, loss=0, neither=5)), 1.0)

    def test_score_statistic_at_zero_is_mcnemar(self):
        for gain, loss in ((3, 9), (10, 4), (1, 1)):
            self.assertAlmostEqual(tango_z(200, gain, loss, 0.0), (gain - loss) / math.sqrt(gain + loss))
        self.assertEqual(tango_z(200, 0, 0, 0.0), 0.0)

    def test_restricted_estimate_maximises_the_constrained_likelihood(self):
        n, gain, loss = 150, 4, 11
        for delta0 in (-0.08, -0.03, 0.0, 0.02):
            def log_likelihood(p_loss):
                p_gain = p_loss + delta0
                rest = 1.0 - p_gain - p_loss
                if p_loss <= 0 or p_gain <= 0 or rest <= 0:
                    return -math.inf
                return gain * math.log(p_gain) + loss * math.log(p_loss) + (n - gain - loss) * math.log(rest)
            grid = max((k / 200000 for k in range(1, 100000)), key=log_likelihood)
            self.assertAlmostEqual(tango_restricted_loss(n, gain, loss, delta0), grid, places=4)

    def test_interval_matches_an_independent_grid_inversion(self):
        # synthetic comparison with a known table: 500 pairs, 5 gains, 8 losses
        counts = PairedCounts(both=440, gain=5, loss=8, neither=47)
        low, high = tango_interval(counts)
        grid_low, grid_high = brute_force_interval(counts, 1.959963984540054)
        self.assertAlmostEqual(low, grid_low, delta=2e-5)
        self.assertAlmostEqual(high, grid_high, delta=2e-5)
        self.assertLess(low, counts.difference)
        self.assertGreater(high, counts.difference)

    def test_noninferiority_decision_is_the_lower_bound_above_the_margin(self):
        for gain, loss in ((5, 8), (2, 14), (0, 0), (9, 3), (1, 20)):
            counts = PairedCounts(both=400, gain=gain, loss=loss, neither=100 - gain - loss)
            result = noninferiority(counts, 0.03, 0.025)
            self.assertEqual(result["noninferior"], result["interval"][0] > -0.03)
        concordant = noninferiority(PairedCounts(both=900, gain=0, loss=0, neither=100), 0.03, 0.025)
        self.assertTrue(concordant["noninferior"])
        self.assertTrue(math.isfinite(concordant["z_at_margin"]))
        lost = noninferiority(PairedCounts(both=400, gain=0, loss=40, neither=60), 0.03, 0.025)
        self.assertFalse(lost["noninferior"])

    def test_counts_from_pairs_orient_gains_and_losses(self):
        counts = PairedCounts.from_pairs([(True, True), (True, False), (False, True), (False, True), (False, False)])
        self.assertEqual((counts.both, counts.loss, counts.gain, counts.neither), (1, 1, 2, 1))
        self.assertAlmostEqual(counts.difference, 1 / 5)
        with self.assertRaises(ValueError):
            PairedCounts.from_pairs([(1, 0)])

    def test_exact_power_and_sample_size(self):
        n = noninferiority_sample_size(0.025, 0.025, 0.03, 0.025, 0.8, step=8)
        self.assertGreaterEqual(noninferiority_power(n, 0.025, 0.025, 0.03, 0.025), 0.8)
        self.assertLess(noninferiority_power(n - 8, 0.025, 0.025, 0.03, 0.025), 0.8)
        normal = noninferiority_sample_size_normal(0.025, 0.025, 0.03, 0.025, 0.8)
        self.assertLess(abs(n - normal) / normal, 0.1)
        self.assertLess(noninferiority_power(200, 0.025, 0.025, 0.03, 0.025),
                        noninferiority_power(600, 0.025, 0.025, 0.03, 0.025))
        # more discordance needs more pairs for the same margin
        self.assertGreater(noninferiority_sample_size(0.04, 0.04, 0.03, 0.025, 0.8, step=8), n)

    def test_size_at_the_margin_is_close_to_alpha(self):
        # true difference exactly -margin (gain 1 %, loss 4 %): rejection rate is the test's size
        for n in (300, 600):
            self.assertLess(noninferiority_power(n, 0.01, 0.04, 0.03, 0.025), 0.035)

    def test_bootstrap_is_seeded_and_brackets_the_estimate(self):
        strata = [[1, 0, 0, -1, 0, 0, 0, -1] * 10, [0, 0, 1, 0, -1, 0] * 10]
        first = paired_bootstrap_interval(strata, 500, 7)
        self.assertEqual(first, paired_bootstrap_interval(strata, 500, 7))
        estimate = sum(map(sum, strata)) / sum(map(len, strata))
        self.assertLessEqual(first[0], estimate)
        self.assertGreaterEqual(first[1], estimate)
        self.assertEqual(paired_bootstrap_interval([[0] * 20], 200, 1), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
