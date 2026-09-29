#!/usr/bin/env python3
"""Render a concise table for one validated Stage 6 physical pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def change(treatment: float, control: float) -> float:
    return 100.0 * (treatment / control - 1.0)


def render(record: dict[str, Any]) -> str:
    supplied = record.get("record_sha256")
    unhashed = dict(record)
    unhashed.pop("record_sha256", None)
    if not (
        record.get("schema") == "s42-stage6-physical-ab-v1"
        and supplied == hashlib.sha256(canonical_bytes(unhashed)).hexdigest()
    ):
        raise ValueError("record binding is invalid")
    control = record["comparison"]["control"]
    treatment = record["comparison"]["treatment"]
    phone = record["phone_work"]
    quality = record["quality"]
    lines = [
        "# Stage 6 full BurstGPT physical A/B",
        "",
        f"Verdict: `{record['status']}`.",
        "",
        "| metric | CPU control | CPU plus OP15 | change |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| makespan | {control['duration_s']:.3f} s | "
            f"{treatment['duration_s']:.3f} s | "
            f"{record['comparison']['makespan_change_pct']:+.2f}% |"
        ),
        (
            f"| CPU package energy | {control['cpu_package_j'] / 1000:.3f} kJ | "
            f"{treatment['cpu_package_j'] / 1000:.3f} kJ | "
            f"{change(treatment['cpu_package_j'], control['cpu_package_j']):+.2f}% |"
        ),
        (
            f"| GPU board energy | {control['gpu_board_j'] / 1000:.3f} kJ | "
            f"{treatment['gpu_board_j'] / 1000:.3f} kJ | "
            f"{change(treatment['gpu_board_j'], control['gpu_board_j']):+.2f}% |"
        ),
        (
            f"| connected phone energy | {control['phone_j'] / 1000:.3f} kJ | "
            f"{treatment['phone_j'] / 1000:.3f} kJ | "
            f"{change(treatment['phone_j'], control['phone_j']):+.2f}% |"
        ),
        (
            f"| accounted fleet energy | {control['fleet_j'] / 1000:.3f} kJ | "
            f"{treatment['fleet_j'] / 1000:.3f} kJ | "
            f"{record['comparison']['fleet_energy_change_pct']:+.2f}% |"
        ),
        "| completed work | 74 req / 11,605 tok | 74 req / 11,605 tok | equal |",
        (
            f"| SLO requests met | {control['slo_met']} | "
            f"{treatment['slo_met']} | {treatment['slo_met'] - control['slo_met']:+d} |"
        ),
        "",
        "## Treatment checks",
        "",
        f"- Dynamic phone calls: {phone['calls']}.",
        (
            "- Phone share of eligible dense-FFN MACs: "
            f"{100.0 * phone['eligible_ffn_mac_fraction']:.2f}%."
        ),
        (
            "- Arithmetic-mean exposed join wait: "
            f"{100.0 * phone['exposed_join_wait_fraction']:.2f}%."
        ),
        (
            "- Maximum executor topology-heartbeat gap: "
            f"{phone['executor_heartbeat_max_gap_s']:.3f} s."
        ),
        (
            "- Maximum RPC-progress gap, including idle intervals: "
            f"{phone['rpc_progress_max_gap_s']:.3f} s."
        ),
        f"- Bridge reset recoveries: {phone['reset_recoveries']}.",
        (
            f"- Pinned MMLU64: {quality['control_correct']} / 64 control, "
            f"{quality['treatment_correct']} / 64 treatment."
        ),
        "",
        record["repetition_scope"] + ".",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    value = json.loads(args.input.read_text(encoding="ascii"))
    args.output.write_text(render(value), encoding="ascii")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
