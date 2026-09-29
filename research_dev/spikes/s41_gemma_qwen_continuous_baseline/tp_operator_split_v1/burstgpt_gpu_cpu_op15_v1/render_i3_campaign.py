#!/usr/bin/env python3
"""Render a validated I3 physical-campaign record as Markdown."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import analyze_i3_campaign
import run_trace


def render(record: dict[str, Any]) -> str:
    supplied = record.get("record_sha256")
    unhashed = dict(record)
    unhashed.pop("record_sha256", None)
    expected = hashlib.sha256(run_trace.canonical(unhashed)).hexdigest()
    run_trace.require(
        record.get("schema") == "s41-i3-physical-campaign-v1" and
        supplied == expected,
        "campaign record binding",
    )
    control = record["headline"]["control"]
    treatment = record["headline"]["treatment"]
    lines = [
        "# I3 real-device BurstGPT energy campaign",
        "",
        f"Verdict: `{record['verdict']}`.",
        "",
        "| average over paired runs | CPU control | CPU plus OP15 | change |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| trace makespan | {control['duration_s']:.3f} s | "
            f"{treatment['duration_s']:.3f} s | "
            f"{record['headline']['makespan_change_pct']:+.2f}% |"
        ),
        (
            f"| CPU package energy | {control['cpu_package_j'] / 1000:.3f} kJ | "
            f"{treatment['cpu_package_j'] / 1000:.3f} kJ | "
            f"{analyze_i3_campaign.percent(treatment['cpu_package_j'], control['cpu_package_j']):+.2f}% |"
        ),
        (
            f"| GPU board energy | {control['gpu_board_j'] / 1000:.3f} kJ | "
            f"{treatment['gpu_board_j'] / 1000:.3f} kJ | "
            f"{analyze_i3_campaign.percent(treatment['gpu_board_j'], control['gpu_board_j']):+.2f}% |"
        ),
        (
            f"| server compute-device energy | {control['server_j'] / 1000:.3f} kJ | "
            f"{treatment['server_j'] / 1000:.3f} kJ | "
            f"{record['headline']['server_energy_change_pct']:+.2f}% |"
        ),
        (
            f"| whole connected phone energy | {control['phone_j'] / 1000:.3f} kJ | "
            f"{treatment['phone_j'] / 1000:.3f} kJ | "
            f"{analyze_i3_campaign.percent(treatment['phone_j'], control['phone_j']):+.2f}% |"
        ),
        (
            f"| accounted fleet energy | {control['fleet_j'] / 1000:.3f} kJ | "
            f"{treatment['fleet_j'] / 1000:.3f} kJ | "
            f"{record['headline']['fleet_energy_change_pct']:+.2f}% |"
        ),
        "| completed work | 74 req / 11,605 tok | 74 req / 11,605 tok | equal |",
        "",
        "## Individual pairs",
        "",
        "| pair | control time | treatment time | time change | control fleet | treatment fleet | fleet change |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair in record["pairs"]:
        lines.append(
            f"| {pair['repeat_index']} | {pair['control']['duration_s']:.3f} s | "
            f"{pair['treatment']['duration_s']:.3f} s | "
            f"{pair['makespan_change_pct']:+.2f}% | "
            f"{pair['control']['fleet_j'] / 1000:.3f} kJ | "
            f"{pair['treatment']['fleet_j'] / 1000:.3f} kJ | "
            f"{pair['fleet_energy_change_pct']:+.2f}% |"
        )
    quality = record["quality"]
    work = record["phone_work"]
    lines.extend([
        "",
        "## Quality and overlap",
        "",
        (
            f"- Pinned MMLU64: {quality['control']['correct']} / 64 control, "
            f"{quality['treatment']['correct']} / 64 treatment."
        ),
        "- Cross-run greedy token equality is diagnostic only because continuous-batch geometry changes between runs.",
        f"- Paid phone calls per treatment: {work['calls']}.",
        f"- Phone MACs per treatment: {work['phone_macs']}.",
        (
            "- Phone share of eligible dense-FFN MACs: "
            f"{100.0 * work['eligible_ffn_mac_fraction']:.2f}%."
        ),
        (
            "- Arithmetic-mean exposed join wait: "
            f"{100.0 * work['mean_exposed_join_wait_fraction']:.2f}%."
        ),
        f"- Bridge reset recoveries: {work['reset_recoveries']}.",
        "",
        "## Energy boundary",
        "",
        f"The headline is {record['boundary']['scope']}.",
        "Accounted components: " + ", ".join(record["boundary"]["accounted"]) + ".",
        "Excluded components: " + ", ".join(record["boundary"]["excluded"]) + ".",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_trace.require(args.output.is_absolute() and not args.output.exists(), "output")
    record = analyze_i3_campaign.load_json(args.input)
    args.output.write_text(render(record), encoding="ascii")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
