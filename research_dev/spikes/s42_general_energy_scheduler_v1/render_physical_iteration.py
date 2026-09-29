#!/usr/bin/env python3
"""Render one validated S42 physical iteration as concise Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from physical_iteration import (
    IterationError,
    _no_duplicates,
    validate_iteration_record,
)


def percent(value: float | None) -> str:
    return "MISSING" if value is None else f"{value:+.2f}%"


def energy(value: dict[str, Any] | None) -> str:
    return "MISSING" if value is None else f"{value['average_j']:.3f} J"


def render(value: object) -> str:
    record = validate_iteration_record(value)
    headline = record["real_device_headline"]
    latency = headline["latency"]
    fleet = headline["accounted_fleet_energy"]
    work = headline["phone_work"]
    overlap = work["overlap_p50_diagnostic"]
    quality = headline["quality"]
    mean_overlap = work["overlap_mean"]
    control_slo = headline["slo_met_average"]["control"]
    treatment_slo = headline["slo_met_average"]["treatment"]
    fleet_energy_text = (
        "missing"
        if fleet["treatment"] is None
        else f"{fleet['treatment']['average_j']:.3f}"
    )
    energy_change_text = (
        "missing"
        if fleet["change_pct"] is None
        else f"{fleet['change_pct']:.2f}"
    )
    lines = [
        f"# Physical iteration {record['iteration_id']}",
        "",
        f"Declared change: {record['declared_one_change']}",
        f"Physical pairs: {record['repetitions']}",
        "",
        "| metric | control | treatment | change |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| BurstGPT makespan | {latency['control_average_s']:.6f} s | "
            f"{latency['treatment_average_s']:.6f} s | "
            f"{percent(latency['change_pct'])} |"
        ),
        (
            f"| accounted fleet energy | {energy(fleet['control'])} | "
            f"{energy(fleet['treatment'])} | {percent(fleet['change_pct'])} |"
        ),
        "| completed work | 74 req / 2175 tok | 74 req / 2175 tok | equal |",
        f"| SLO requests met | {control_slo:.1f} | {treatment_slo:.1f} | {treatment_slo - control_slo:+.1f} |",
        f"| phone-routed requests | 0 / 74 | {work['phone_requests']} / 74 | +{work['phone_requests']} |",
        (
            "| phone share of eligible cold FFN MACs | 0% | "
            f"{100.0 * work['phone_fraction_eligible_cold_ffn_macs']:.2f}% | "
            f"+{100.0 * work['phone_fraction_eligible_cold_ffn_macs']:.2f} points |"
        ),
        "",
        "## Phone work",
        "",
        f"- Paid calls excluding warmup: {work['paid_phone_calls']}",
        f"- Phone MACs: {work['phone_macs']}",
        f"- Eligible cold FFN MACs: {work['eligible_cold_ffn_macs']}",
        f"- Host-to-phone paid bytes: {work['paid_upload_bytes']}",
        f"- Phone-to-host paid bytes: {work['paid_download_bytes']}",
        (
            f"- p50 branch imbalance: "
            f"{100.0 * overlap['branch_imbalance_fraction']:.2f}%"
        ),
        (
            f"- p50 exposed join wait: "
            f"{100.0 * overlap['exposed_join_wait_fraction']:.2f}% of island"
        ),
        (
            f"- p50 phone path hidden: "
            f"{100.0 * overlap['phone_work_hidden_fraction']:.2f}%"
        ),
        (
            "- Arithmetic-mean exposed join wait: MISSING"
            if mean_overlap["exposed_join_wait_fraction"] is None
            else (
                "- Arithmetic-mean exposed join wait: "
                f"{100.0 * mean_overlap['exposed_join_wait_fraction']:.2f}% of island"
            )
        ),
        f"- Exact cold outputs: {quality['cold_requests_exact']} / {quality['cold_requests_total']}",
        (
            f"- Equal cold token positions: {quality['cold_token_positions_equal']} / "
            f"{quality['cold_token_positions_total']}"
        ),
        "",
        f"VERDICT: {'INCOMPLETE' if record['verdict'].startswith('INCOMPLETE') else record['verdict']}",
        (
            "BEST_REAL_RESULT: "
            f"{latency['treatment_average_s']:.6f} s, "
            f"accounted_energy_j={fleet_energy_text}, "
            f"energy_change_pct={energy_change_text}, "
            f"phone_work_pct={100.0 * work['phone_fraction_eligible_cold_ffn_macs']:.2f}"
        ),
        f"BLOCKER: {record['blocker']}",
        f"NEXT_ONE_CHANGE: {record['next_one_change']}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        with args.input.open("r", encoding="ascii") as source:
            value = json.load(source, object_pairs_hook=_no_duplicates)
        text = render(value)
    except (OSError, UnicodeError, json.JSONDecodeError, IterationError) as exc:
        parser.exit(2, f"render failed: {exc}\n")
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="ascii")
        print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
