"""Paired-binary statistics for the quality noninferiority test (standard library only).

Orientation everywhere: d = p_T - p_B, treatment accuracy minus baseline accuracy. For one item scored in
both arms, a *gain* is treatment correct / baseline wrong and a *loss* is treatment wrong / baseline correct.
Noninferiority with margin M (a positive proportion, e.g. 0.03) rejects H0: d <= -M.

Primary test: Tango's asymptotic score test and its inverted two-sided confidence interval (Tango 1998,
Stat Med 17:891-908; recommended for paired proportions by Fagerland, Lydersen and Laake 2013, BMC Med Res
Methodol 13:91). Sensitivity: Newcombe's hybrid score interval (method 10, Newcombe 1998, Stat Med
17:2635-2650), a stratified paired bootstrap, and the exact (conditional) McNemar test of d = 0.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import NormalDist


def z_quantile(probability: float) -> float:
    return NormalDist().inv_cdf(probability)


@dataclass(frozen=True)
class PairedCounts:
    """2x2 table of one comparison: rows treatment correct/wrong, columns baseline correct/wrong."""

    both: int
    loss: int
    gain: int
    neither: int

    def __post_init__(self) -> None:
        for name in ("both", "loss", "gain", "neither"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"paired count {name} is invalid")
        if self.n == 0:
            raise ValueError("paired counts are empty")

    @classmethod
    def from_pairs(cls, pairs) -> "PairedCounts":
        """pairs: iterable of (baseline_correct, treatment_correct) booleans."""
        cells = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
        for baseline, treatment in pairs:
            if type(baseline) is not bool or type(treatment) is not bool:
                raise ValueError("paired outcome is not boolean")
            cells[(baseline, treatment)] += 1
        return cls(both=cells[(True, True)], loss=cells[(True, False)],
                   gain=cells[(False, True)], neither=cells[(False, False)])

    @property
    def n(self) -> int:
        return self.both + self.loss + self.gain + self.neither

    @property
    def discordant(self) -> int:
        return self.loss + self.gain

    @property
    def baseline_accuracy(self) -> float:
        return (self.both + self.loss) / self.n

    @property
    def treatment_accuracy(self) -> float:
        return (self.both + self.gain) / self.n

    @property
    def difference(self) -> float:
        return (self.gain - self.loss) / self.n

    def to_json(self) -> dict[str, object]:
        return {"both_correct": self.both, "loss": self.loss, "gain": self.gain, "neither_correct": self.neither,
                "n": self.n, "discordant": self.discordant,
                "baseline_accuracy": self.baseline_accuracy, "treatment_accuracy": self.treatment_accuracy,
                "difference": self.difference}


def tango_restricted_loss(n: float, gain: float, loss: float, delta0: float) -> float:
    """Restricted MLE of P(loss) under d = delta0 (closed-form root of the score equation)."""
    a = 2.0 * n
    b = -gain - loss + (2.0 * n - gain + loss) * delta0
    c = -loss * delta0 * (1.0 - delta0)
    return (math.sqrt(max(0.0, b * b - 4.0 * a * c)) - b) / (2.0 * a)


def tango_z(n: float, gain: float, loss: float, delta0: float) -> float:
    """Tango's score statistic for H: d = delta0; large positive values favour d > delta0."""
    restricted = tango_restricted_loss(n, gain, loss, delta0)
    variance = n * (2.0 * restricted + delta0 * (1.0 - delta0))
    numerator = gain - loss - n * delta0
    if variance <= 1e-300:
        return 0.0 if abs(numerator) < 1e-12 else math.copysign(math.inf, numerator)
    return numerator / math.sqrt(variance)


def _bisect(function, low: float, high: float, iterations: int = 200) -> float:
    """Root of a decreasing function on [low, high] (function(low) >= 0 >= function(high))."""
    for _ in range(iterations):
        middle = (low + high) / 2.0
        if function(middle) >= 0.0:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def tango_interval(counts: PairedCounts, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided score interval {delta0 : |Z(delta0)| <= z}; Z is decreasing in delta0."""
    z = z_quantile(1.0 - (1.0 - confidence) / 2.0)
    n, gain, loss = counts.n, counts.gain, counts.loss
    estimate = counts.difference
    edge = 1.0 - 1e-12
    lower = -1.0 if tango_z(n, gain, loss, -edge) < z else _bisect(
        lambda value: tango_z(n, gain, loss, value) - z, -edge, estimate)
    upper = 1.0 if tango_z(n, gain, loss, edge) > -z else _bisect(
        lambda value: tango_z(n, gain, loss, value) + z, estimate, edge)
    return lower, upper


def wilson_interval(successes: int, n: int, z: float) -> tuple[float, float]:
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_interval(counts: PairedCounts, confidence: float = 0.95) -> tuple[float, float]:
    """Newcombe's hybrid score interval for the paired difference (method 10, no continuity correction)."""
    z = z_quantile(1.0 - (1.0 - confidence) / 2.0)
    n = counts.n
    treatment_correct = counts.both + counts.gain
    baseline_correct = counts.both + counts.loss
    p_t, p_b = treatment_correct / n, baseline_correct / n
    l_t, u_t = wilson_interval(treatment_correct, n, z)
    l_b, u_b = wilson_interval(baseline_correct, n, z)
    margins = (treatment_correct * (n - treatment_correct) * baseline_correct * (n - baseline_correct))
    # Newcombe's corrected phi: a positive association is shrunk by N/2 (floored at zero)
    association = counts.both * counts.neither - counts.gain * counts.loss
    if association > 0:
        association = max(association - n / 2.0, 0.0)
    phi = 0.0 if margins == 0 else association / math.sqrt(margins)
    difference = p_t - p_b
    lower = difference - math.sqrt(max(0.0, (p_t - l_t) ** 2 - 2.0 * phi * (p_t - l_t) * (u_b - p_b)
                                       + (u_b - p_b) ** 2))
    upper = difference + math.sqrt(max(0.0, (u_t - p_t) ** 2 - 2.0 * phi * (u_t - p_t) * (p_b - l_b)
                                       + (p_b - l_b) ** 2))
    return max(-1.0, lower), min(1.0, upper)


def _log_binomial(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log(1.0 - p))


def binomial_pmf(k: int, n: int, p: float) -> float:
    return math.exp(_log_binomial(k, n, p))


def exact_mcnemar_p(counts: PairedCounts) -> float:
    """Two-sided exact conditional McNemar p-value for d = 0 (binomial test on the discordant pairs)."""
    m = counts.discordant
    if m == 0:
        return 1.0
    tail = sum(binomial_pmf(k, m, 0.5) for k in range(0, min(counts.gain, counts.loss) + 1))
    return min(1.0, 2.0 * tail)


def paired_bootstrap_interval(strata: list[list[int]], replicates: int, seed: int,
                              confidence: float = 0.95) -> tuple[float, float]:
    """Percentile interval of the pooled mean paired difference, resampling items within each stratum.

    strata: per-stratum lists of per-item differences in {-1, 0, 1}; the pooled estimate weights strata by
    their (fixed) sizes, as the design does."""
    if replicates < 1 or not strata or not all(strata):
        raise ValueError("bootstrap input is invalid")
    total_n = sum(len(stratum) for stratum in strata)
    generator = random.Random(seed)
    means = []
    for _ in range(replicates):
        total = 0
        for stratum in strata:
            total += sum(generator.choices(stratum, k=len(stratum)))
        means.append(total / total_n)
    means.sort()
    tail = (1.0 - confidence) / 2.0
    low_index = min(replicates - 1, math.floor(tail * replicates))
    high_index = max(0, math.ceil((1.0 - tail) * replicates) - 1)
    return means[low_index], means[high_index]


def noninferiority(counts: PairedCounts, margin: float, alpha: float) -> dict[str, object]:
    """Tango score test of H0: d <= -margin at one-sided level alpha, with its 1 - 2 alpha interval."""
    if not 0.0 < margin < 1.0 or not 0.0 < alpha < 0.5:
        raise ValueError("noninferiority parameters are invalid")
    z_boundary = tango_z(counts.n, counts.gain, counts.loss, -margin)
    lower, upper = tango_interval(counts, 1.0 - 2.0 * alpha)
    critical = z_quantile(1.0 - alpha)
    return {
        "method": "tango-score",
        "margin": margin,
        "alpha_one_sided": alpha,
        "estimate": counts.difference,
        "interval": [lower, upper],
        "interval_confidence": 1.0 - 2.0 * alpha,
        "z_at_margin": z_boundary,
        "critical_z": critical,
        "p_value_one_sided": 1.0 - NormalDist().cdf(z_boundary) if math.isfinite(z_boundary) else (
            0.0 if z_boundary > 0 else 1.0),
        "noninferior": bool(z_boundary > critical),
    }


def noninferiority_power(n: int, p_gain: float, p_loss: float, margin: float, alpha: float) -> float:
    """Exact power of the Tango noninferiority test: sum over the trinomial of (gains, losses).

    The discordant total m ~ Bin(n, p_gain + p_loss), gains | m ~ Bin(m, p_gain / (p_gain + p_loss))."""
    if n < 1 or p_gain < 0.0 or p_loss < 0.0 or p_gain + p_loss > 1.0:
        raise ValueError("power parameters are invalid")
    critical = z_quantile(1.0 - alpha)
    discordance = p_gain + p_loss
    if discordance == 0.0:
        return 1.0 if tango_z(n, 0, 0, -margin) > critical else 0.0
    share = p_gain / discordance
    mean = n * discordance
    power = 0.0
    for m in range(0, n + 1):
        weight = binomial_pmf(m, n, discordance)
        if m > mean and weight < 1e-15:
            break
        if weight < 1e-18:
            continue
        power += weight * sum(binomial_pmf(g, m, share) for g in range(m + 1)
                              if tango_z(n, g, m - g, -margin) > critical)
    return power


def noninferiority_sample_size_normal(p_gain: float, p_loss: float, margin: float, alpha: float,
                                      power: float) -> int:
    """Asymptotic sample size of the Tango test (Nam 1997 / Tango 1998 form), for cross-checking."""
    discordance = p_gain + p_loss
    true_difference = p_gain - p_loss
    delta0 = -margin
    b = -discordance + (2.0 - p_gain + p_loss) * delta0
    c = -p_loss * delta0 * (1.0 - delta0)
    restricted = (math.sqrt(max(0.0, b * b - 8.0 * c)) - b) / 4.0
    numerator = (z_quantile(1.0 - alpha) * math.sqrt(2.0 * restricted + delta0 * (1.0 - delta0))
                 + z_quantile(power) * math.sqrt(max(0.0, discordance - true_difference ** 2)))
    return math.ceil((numerator / (true_difference - delta0)) ** 2)


def noninferiority_sample_size(p_gain: float, p_loss: float, margin: float, alpha: float,
                               power: float, step: int = 1) -> int:
    """Smallest n (on a grid of `step`) whose exact power reaches `power`, searched upward from 60 % of the
    asymptotic answer."""
    start = max(step, int(0.6 * noninferiority_sample_size_normal(p_gain, p_loss, margin, alpha, power)))
    start -= start % step
    n = max(step, start)
    while noninferiority_power(n, p_gain, p_loss, margin, alpha) < power:
        n += step
        if n > 1_000_000:
            raise ValueError("sample size search did not converge")
    return n
