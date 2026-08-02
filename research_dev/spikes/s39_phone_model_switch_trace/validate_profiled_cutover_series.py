#!/usr/bin/env python3
"""Rebuild and validate the four-pair W9 certificate from raw evidence."""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import phone_cuda_delta_probe as w6
import w9_profiled_cutover as w9
from validate_profiled_cutover_pair import validate_pair


SCHEMA = "s39-profiled-cutover-series-certificate-v1"


def fraction_value(value: Fraction) -> dict[str, int]:
    return {"denominator": value.denominator, "numerator": value.numerator}


def even_median(values: list[Fraction]) -> Fraction:
    w9.require(len(values) == 4, "series: expected four values")
    ordered = sorted(values)
    return (ordered[1] + ordered[2]) / 2


def validate_series(
    series_dir: Path,
    *,
    contract_path: Path,
    w8_contract_path: Path,
    base_contract_path: Path,
    delta_contract_path: Path,
    physical_gate_path: Path,
) -> dict[str, object]:
    contract = w9.load_contract(contract_path)
    context, context_raw = w6.read_canonical(
        series_dir / "CAMPAIGN_CONTEXT.json",
        "campaign_context",
    )
    root = w9.exact_keys(
        context,
        {
            "base_git_commit",
            "contract_sha256",
            "created_utc_ns",
            "pair_ordinals",
            "schema",
        },
        "campaign_context",
    )
    w9.require(
        root["schema"] == "s39-profiled-cutover-campaign-context-v1"
        and root["contract_sha256"] == contract.raw_sha256
        and root["pair_ordinals"] == list(contract.pair_ordinals)
        and type(root["base_git_commit"]) is str
        and len(root["base_git_commit"]) == 40
        and w9.is_int(root["created_utc_ns"])
        and root["created_utc_ns"] > 0,
        "series: campaign context",
    )
    directories = sorted(
        item.name
        for item in series_dir.iterdir()
        if item.is_dir() and item.name.startswith("P")
    )
    w9.require(
        directories == list(contract.pair_ordinals),
        "series: paid pair set or replacement",
    )

    pairs = []
    run_ids = set()
    for ordinal in contract.pair_ordinals:
        pair_dir = series_dir / ordinal
        paid_marker = pair_dir / "treatment/start.json"
        w9.require(paid_marker.is_file(), f"series: {ordinal} has no paid marker")
        rebuilt = validate_pair(
            pair_dir,
            contract_path=contract_path,
            w8_contract_path=w8_contract_path,
            base_contract_path=base_contract_path,
            delta_contract_path=delta_contract_path,
            physical_gate_path=physical_gate_path,
        )
        stored, stored_raw = w6.read_canonical(
            pair_dir / "pair_certificate.json",
            f"series.{ordinal}.certificate",
        )
        w9.require(
            stored == rebuilt
            and stored_raw == w6.canonical(rebuilt)
            and stored["status"] == "PROFILED_ZERO_EXTRA_PAIR_PASS"
            and stored["pair_ordinal"] == ordinal
            and stored["run_id"] not in run_ids,
            f"series: {ordinal} certificate mismatch",
        )
        run_ids.add(stored["run_id"])
        pairs.append(stored)

    next_ratios = [
        Fraction(
            pair["metrics"]["treatment_promotion_next_token_us"],
            pair["metrics"]["control_promotion_next_token_us"],
        )
        for pair in pairs
    ]
    completion_ratios = [
        Fraction(
            pair["metrics"]["treatment_completion_us"],
            pair["metrics"]["control_completion_us"],
        )
        for pair in pairs
    ]
    next_median = even_median(next_ratios)
    completion_median = even_median(completion_ratios)
    next_gate = Fraction(
        contract.next_token_ratio_num,
        contract.next_token_ratio_den,
    )
    completion_gate = Fraction(
        contract.completion_ratio_num,
        contract.completion_ratio_den,
    )
    w9.require(
        next_median <= next_gate,
        "series: promotion-next-token ratio gate",
    )
    w9.require(
        completion_median <= completion_gate,
        "series: completion ratio gate",
    )
    w9.require(
        any(pair["diagnostics"]["d_actual"] == 1 for pair in pairs),
        "series: in-flight branch was not exercised",
    )
    thermal_ranges = {}
    for device in ("op12", "op15"):
        starts = [pair["thermal_start_millic"][device] for pair in pairs]
        span = max(starts) - min(starts)
        w9.require(
            span <= contract.phone_thermal_range_millic,
            f"series: {device} treatment-start thermal range",
        )
        thermal_ranges[device] = {
            "maximum_millic": max(starts),
            "minimum_millic": min(starts),
            "range_millic": span,
            "values_millic": starts,
        }
    gpu_identity = {
        (
            pair["diagnostics"]["gpu_name"],
            pair["placement"]["treatment_cuda_head"]["primary_buffer"],
        )
        for pair in pairs
    }
    w9.require(len(gpu_identity) == 1, "series: CUDA identity changed")

    return {
        "campaign_context_sha256": w6.sha256(context_raw),
        "contract_sha256": contract.raw_sha256,
        "diagnostics": {
            "all_pairs_valid": True,
            "greedy_matching_tokens": [
                pair["diagnostics"]["control_greedy_matching_tokens"]
                for pair in pairs
            ],
            "greedy_total_tokens": [
                pair["diagnostics"]["control_greedy_total_tokens"]
                for pair in pairs
            ],
            "inflight_pairs": [
                pair["pair_ordinal"]
                for pair in pairs
                if pair["diagnostics"]["d_actual"] == 1
            ],
        },
        "pair_certificates": [
            {
                "pair_ordinal": pair["pair_ordinal"],
                "sha256": w6.sha256(
                    (series_dir / pair["pair_ordinal"] / "pair_certificate.json")
                    .read_bytes()
                ),
            }
            for pair in pairs
        ],
        "pair_metrics": [
            {
                "completion_ratio": fraction_value(ratio_completion),
                "control_completion_us": pair["metrics"][
                    "control_completion_us"
                ],
                "control_promotion_next_token_us": pair["metrics"][
                    "control_promotion_next_token_us"
                ],
                "next_token_ratio": fraction_value(ratio_next),
                "pair_ordinal": pair["pair_ordinal"],
                "treatment_completion_us": pair["metrics"][
                    "treatment_completion_us"
                ],
                "treatment_promotion_next_token_us": pair["metrics"][
                    "treatment_promotion_next_token_us"
                ],
            }
            for pair, ratio_next, ratio_completion in zip(
                pairs,
                next_ratios,
                completion_ratios,
            )
        ],
        "performance": {
            "completion_gate": fraction_value(completion_gate),
            "completion_median_ratio": fraction_value(completion_median),
            "next_token_gate": fraction_value(next_gate),
            "next_token_median_ratio": fraction_value(next_median),
        },
        "scheduler_eligible": False,
        "schema": SCHEMA,
        "scope": "MECHANICS_ONLY",
        "status": "PROFILED_ZERO_EXTRA_CUTOVER_MECHANICS_PASS",
        "thermal_start_ranges": thermal_ranges,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--w8-contract", type=Path, required=True)
    parser.add_argument("--base-contract", type=Path, required=True)
    parser.add_argument("--delta-contract", type=Path, required=True)
    parser.add_argument("--physical-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    certificate = validate_series(
        args.series_dir,
        contract_path=args.contract,
        w8_contract_path=args.w8_contract,
        base_contract_path=args.base_contract,
        delta_contract_path=args.delta_contract,
        physical_gate_path=args.physical_gate,
    )
    w9.write_atomic(args.output, certificate)
    print(w6.canonical(certificate).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError, w6.DeltaError, w9.W9Error) as exc:
        print(w6.canonical({
            "error": str(exc),
            "status": "PROFILED_ZERO_EXTRA_SERIES_VALIDATION_ERROR",
        }).decode("ascii"), end="")
        raise SystemExit(2)
