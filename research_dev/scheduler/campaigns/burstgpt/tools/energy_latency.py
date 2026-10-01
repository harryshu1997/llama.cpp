#!/usr/bin/env python3
"""Energy versus latency of a set of arms, and energy at matched latency (read-only).

    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.energy_latency \
        --run legacy-r1=DIR:legacy --run legacy-r2=DIR:legacy --run s2a=DIR:phones ... \
        --reference legacy --reference dispatcher --treatment phones [--frozen s2a] [--json OUT] [--md OUT]

Every run is one point: energy (host = CPU package + GPU board; fleet = host + both phones' measured terminal
energy, from fleet_energy.py) against two latency metrics from latency_report.py: p90 arrival-to-completion (nearest
rank) and mean TPOT (mean over requests of (end - first token) / (output tokens - 1)).

Matched latency, stated method (never extrapolated):
  step  E_F(L) = min energy over family F's runs with latency <= L. No interpolation: every value is a run that
        was measured and is at least as fast as L. Valid for any latency metric, including p90.
  hull  piecewise-linear lower convex hull of F's (latency, energy) points; a target faster than every run is
        "not attainable" (never extrapolated), a target slower than every run takes the cheapest run. Between two
        configurations it assumes a time-shared mix of them reaches the chord; that is
        defensible for trace-additive quantities (energy, mean TPOT) and is NOT for a percentile: hull values at a
        p90 target are labelled ``assumption: mix``.
  For every reference run R and treatment family T the comparison is taken at R's own latency L_R: saving =
  1 - E_T(L_R) / E_R. ``dominates`` lists the treatment runs with lower energy AND lower-or-equal latency than R.

Honesty about single runs: a family made of different configurations with one run each turns its frontier into a
best-of-N over noisy runs (optimistic). The report therefore also gives (a) the frozen configuration alone
(``--frozen``, a single run: dominance only), and (b) the run-to-run spread of every configuration that was
repeated (``--repeat label1,label2``), which bounds the differences that single runs can resolve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

from research_dev.scheduler.campaigns.burstgpt.tools.fleet_energy import account_run
from research_dev.scheduler.campaigns.burstgpt.tools.latency_report import report_run

SCHEMA = "ws3-energy-latency-v1"
LATENCY_METRICS = {"p90_e2e_s": "p90 arrival-to-completion (s, nearest rank)",
                   "mean_tpot_ms": "mean TPOT over requests (ms)"}
ENERGY_METRICS = ("host_kj", "fleet_kj")
PERCENTILE_METRICS = {"p90_e2e_s"}


def point(label: str, family: str, directory: Path) -> dict[str, Any]:
    energy = account_run(label, directory)
    latency = report_run(label, directory)
    overall = latency["overall"]
    return {"label": label, "family": family, "duration_s": energy["duration_s"],
            "host_kj": energy["desktop"]["host_j"] / 1e3,
            "fleet_kj": None if energy["fleet_j"] is None else energy["fleet_j"] / 1e3,
            "phones_kj": None if energy["phones_j"] is None else energy["phones_j"] / 1e3,
            "p90_e2e_s": overall["e2e_s"]["p90"], "p50_e2e_s": overall["e2e_s"]["p50"],
            "mean_e2e_s": overall["e2e_s"]["mean"], "mean_tpot_ms": overall["tpot_ms"]["mean"],
            "token_weighted_tpot_ms": overall.get("token_weighted_tpot_ms"),
            "p90_ttft_s": overall["ttft_s"]["p90"]}


def step_energy(points: Sequence[Mapping[str, Any]], latency: str, energy: str, target: float) -> dict[str, Any] | None:
    eligible = [p for p in points if p[latency] is not None and p[energy] is not None and p[latency] <= target + 1e-9]
    if not eligible:
        return None
    best = min(eligible, key=lambda p: (p[energy], p[latency]))
    return {"energy": best[energy], "run": best["label"], "latency": best[latency]}


def lower_hull(xy: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Lower convex hull (monotone chain) of points sorted by x, keeping the minimum y per x."""
    best: dict[float, float] = {}
    for x, y in xy:
        best[x] = min(y, best.get(x, y))
    hull: list[tuple[float, float]] = []
    for x, y in sorted(best.items()):
        while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (y - hull[-2][1]) - (hull[-1][1] - hull[-2][1]) * (x - hull[-2][0]) <= 0:
            hull.pop()
        hull.append((x, y))
    return hull


def hull_energy(points: Sequence[Mapping[str, Any]], latency: str, energy: str, target: float) -> dict[str, Any] | None:
    hull = lower_hull([(p[latency], p[energy]) for p in points if p[latency] is not None and p[energy] is not None])
    if not hull or target < hull[0][0] - 1e-9:
        return None  # faster than every measured run: not attainable, never extrapolated
    # the energy-minimal part of the hull is non-increasing in latency up to its minimum; beyond it, stay flat
    # (a slower target can always be met by the faster, cheaper point: min over the hull left of target)
    values = []
    for (x0, y0), (x1, y1) in zip(hull, hull[1:]):
        if x0 - 1e-9 <= target <= x1 + 1e-9:
            values.append(y0 if x1 == x0 else y0 + (y1 - y0) * (target - x0) / (x1 - x0))
    values.extend(y for x, y in hull if x <= target + 1e-9)
    return {"energy": min(values), "hull": hull,
            "assumption": "mix" if latency in PERCENTILE_METRICS else "time-shared mix (additive metric)"}


def compare(points: Sequence[Mapping[str, Any]], references: Sequence[str], treatments: Sequence[str],
            frozen: str | None) -> list[dict[str, Any]]:
    rows = []
    for reference in [p for p in points if p["family"] in references]:
        for latency in LATENCY_METRICS:
            for energy in ENERGY_METRICS:
                target, base = reference[latency], reference[energy]
                if target is None or base is None:
                    continue
                for family in treatments:
                    members = [p for p in points if p["family"] == family]
                    step = step_energy(members, latency, energy, target)
                    hull = hull_energy(members, latency, energy, target)
                    dominates = sorted(p["label"] for p in members if p[energy] is not None and p[latency] is not None
                                       and p[energy] < base and p[latency] <= target)
                    row = {"reference": reference["label"], "reference_family": reference["family"],
                           "treatment_family": family, "latency_metric": latency, "energy_metric": energy,
                           "target_latency": target, "reference_energy": base,
                           "step": step, "step_saving": None if step is None else 1 - step["energy"] / base,
                           "hull_energy": None if hull is None else hull["energy"],
                           "hull_saving": None if hull is None else 1 - hull["energy"] / base,
                           "hull_assumption": None if hull is None else hull["assumption"],
                           "dominating_runs": dominates}
                    if frozen:
                        pinned = next((p for p in members if p["label"] == frozen), None)
                        if pinned is not None:
                            row["frozen"] = {"run": frozen, "energy": pinned[energy], "latency": pinned[latency],
                                             "no_slower": pinned[latency] <= target,
                                             "saving": 1 - pinned[energy] / base}
                    rows.append(row)
    return rows


def spread(points: Sequence[Mapping[str, Any]], groups: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    out = []
    by_label = {p["label"]: p for p in points}
    for group in groups:
        members = [by_label[label] for label in group if label in by_label]
        if len(members) < 2:
            continue
        row = {"runs": [p["label"] for p in members]}
        for key in ENERGY_METRICS + tuple(LATENCY_METRICS):
            values = [p[key] for p in members if p[key] is not None]
            if len(values) >= 2:
                mean = statistics.fmean(values)
                row[key] = {"min": min(values), "max": max(values), "mean": mean,
                            "range_over_mean": (max(values) - min(values)) / mean if mean else None}
        out.append(row)
    return out


def _f(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else ("%." + str(digits) + "f") % value


def _pct(value: float | None) -> str:
    return "-" if value is None else "%.1f %%" % (100 * value)


def markdown(points: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]],
             spreads: Sequence[Mapping[str, Any]]) -> str:
    lines = ["| run | family | host kJ | fleet kJ | p90 E2E s | p50 E2E s | mean TPOT ms | token-weighted TPOT ms |",
             "|" + "---|" * 8]
    for p in sorted(points, key=lambda p: (p["family"], p["label"])):
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            p["label"], p["family"], _f(p["host_kj"]), _f(p.get("fleet_kj")), _f(p.get("p90_e2e_s"), 0),
            _f(p.get("p50_e2e_s"), 0), _f(p.get("mean_tpot_ms"), 0), _f(p.get("token_weighted_tpot_ms"), 0)))
    lines += ["", "Energy at the reference run's own latency (saving = 1 - E_treatment / E_reference):", "",
              "| reference | latency metric | target | energy | ref kJ | treatment | step kJ (run) | step saving |"
              " hull kJ | hull saving | frozen (no slower?) saving | dominating runs |", "|" + "---|" * 12]
    for row in rows:
        step = row["step"]
        frozen = row.get("frozen")
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            row["reference"], row["latency_metric"], _f(row["target_latency"], 0), row["energy_metric"],
            _f(row["reference_energy"]), row["treatment_family"],
            "-" if step is None else "%.1f (%s)" % (step["energy"], step["run"]), _pct(row["step_saving"]),
            _f(row["hull_energy"]), _pct(row["hull_saving"]) + ("" if row["hull_assumption"] != "mix" else " [mix]"),
            "-" if not frozen else "%s %s" % ("yes" if frozen["no_slower"] else "NO", _pct(frozen["saving"])),
            ", ".join(row["dominating_runs"]) or "none"))
    if spreads:
        lines += ["", "Run-to-run spread of repeated configurations ((max - min) / mean):", "",
                  "| runs | host kJ | fleet kJ | p90 E2E | mean TPOT |", "|---|---|---|---|---|"]
        for row in spreads:
            lines.append("| %s | %s | %s | %s | %s |" % (", ".join(row["runs"]), *(
                _pct((row.get(key) or {}).get("range_over_mean")) for key in ("host_kj", "fleet_kj", "p90_e2e_s", "mean_tpot_ms"))))
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="label=DIR:family")
    parser.add_argument("--reference", action="append", required=True, help="reference family (e.g. legacy)")
    parser.add_argument("--treatment", action="append", required=True, help="treatment family")
    parser.add_argument("--frozen", help="label of the frozen configuration's run")
    parser.add_argument("--repeat", action="append", default=[], help="comma-separated labels of one repeated config")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--md", type=Path)
    args = parser.parse_args(argv)
    points = []
    for spec in args.run:
        label, sep, rest = spec.partition("=")
        directory, colon, family = rest.rpartition(":")
        if not sep or not colon or not family:
            raise SystemExit("expected label=DIR:family, got " + spec)
        points.append(point(label, family, Path(directory)))
    rows = compare(points, args.reference, args.treatment, args.frozen)
    spreads = spread(points, [group.split(",") for group in args.repeat])
    text = markdown(points, rows, spreads)
    if args.json:
        args.json.write_text(json.dumps({"schema": SCHEMA, "latency_metrics": LATENCY_METRICS, "points": points,
                                         "comparisons": rows, "repeat_spread": spreads}, indent=1, sort_keys=True) + "\n")
    if args.md:
        args.md.write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
