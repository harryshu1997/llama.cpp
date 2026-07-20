#!/usr/bin/env python3
"""Compute a two-state GPU-board energy feasibility bound from S11 timing."""

import argparse
import hashlib
import json
import pathlib
import sys
from decimal import Decimal, localcontext
from fractions import Fraction


class AnalysisError(RuntimeError):
    pass


SUPPORTED_SCHEMAS = {
    "s11-fixed-route-poc-result-v2",
    "s11-fixed-route-poc-result-v3",
}
REQUIRED_TREATMENT_ROUTE = "A0_OP15"


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AnalysisError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _positive_int(value, field):
    if type(value) is not int or value <= 0:
        raise AnalysisError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value, field):
    if type(value) is not int or value < 0:
        raise AnalysisError(f"{field} must be a nonnegative integer")
    return value


def _reject_json_constant(value):
    raise AnalysisError(f"invalid JSON constant: {value}")


def _fraction_record(value):
    with localcontext() as context:
        context.prec = 40
        decimal = Decimal(value.numerator) / Decimal(value.denominator)
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": f"{decimal:.9f}",
    }


def extract_timing_rows(summary):
    if type(summary) is not dict:
        raise AnalysisError("summary must be an object")
    if summary.get("schema") not in SUPPORTED_SCHEMAS:
        raise AnalysisError("summary schema is not supported")
    if summary.get("treatment_route") != REQUIRED_TREATMENT_ROUTE:
        raise AnalysisError("summary treatment route is not A0_OP15")
    aggregate = summary.get("aggregate")
    if type(aggregate) is not dict or aggregate.get("exact_work_all_pairs") is not True:
        raise AnalysisError("summary aggregate is not exact")
    pairs = summary.get("pairs")
    if type(pairs) is not list or not pairs:
        raise AnalysisError("summary must contain at least one pair")

    rows = []
    pair_indexes = []
    for pair_position, pair in enumerate(pairs):
        if type(pair) is not dict or pair.get("exact_work") is not True:
            raise AnalysisError(f"pair {pair_position} is not exact")
        pair_index = pair.get("pair_index")
        if type(pair_index) is not int or pair_index < 0:
            raise AnalysisError(f"pair {pair_position} has an invalid index")
        pair_indexes.append(pair_index)
        if pair.get("treatment_route") != REQUIRED_TREATMENT_ROUTE:
            raise AnalysisError(f"pair {pair_index} treatment route is not A0_OP15")
        control = pair.get("control_batch_metrics")
        treatment = pair.get("treatment_batch_metrics")
        if type(control) is not dict or type(treatment) is not dict:
            raise AnalysisError(f"pair {pair_index} lacks batch metrics")
        control_batch = _positive_int(control.get("batch_size"), "control batch_size")
        treatment_batch = _positive_int(
            treatment.get("batch_size"), "treatment batch_size")
        if control_batch != treatment_batch:
            raise AnalysisError(f"pair {pair_index} mixes batch sizes")
        control_groups = control.get("group_records")
        treatment_groups = treatment.get("group_records")
        if (type(control_groups) is not list or
                type(treatment_groups) is not list or
                len(control_groups) != len(treatment_groups) or
                not control_groups):
            raise AnalysisError(f"pair {pair_index} has unmatched groups")

        indexed_groups = []
        for side, groups in (("control", control_groups), ("treatment", treatment_groups)):
            indexed = {}
            for group_position, group in enumerate(groups):
                if type(group) is not dict:
                    raise AnalysisError(
                        f"pair {pair_index} {side} group {group_position} is not an object")
                batch_index = group.get("batch_index")
                if type(batch_index) is not int or batch_index < 0:
                    raise AnalysisError(
                        f"pair {pair_index} {side} group has an invalid batch_index")
                if batch_index in indexed:
                    raise AnalysisError(
                        f"pair {pair_index} {side} duplicates batch_index {batch_index}")
                indexed[batch_index] = group
            indexed_groups.append(indexed)
        control_by_index, treatment_by_index = indexed_groups
        if set(control_by_index) != set(treatment_by_index):
            raise AnalysisError(f"pair {pair_index} has unmatched batch indexes")

        for batch_index in sorted(control_by_index):
            control_group = control_by_index[batch_index]
            treatment_group = treatment_by_index[batch_index]
            if type(control_group) is not dict or type(treatment_group) is not dict:
                raise AnalysisError(
                    f"pair {pair_index} batch {batch_index} is not an object")
            control_wall_us = _positive_int(
                control_group.get("request_wall_us"), "control request_wall_us")
            treatment_wall_us = _positive_int(
                treatment_group.get("request_wall_us"), "treatment request_wall_us")
            treatment_tail_us = _positive_int(
                treatment_group.get("host_us"), "treatment host_us")
            stage_a_us = _nonnegative_int(
                treatment_group.get("stage_a_us"), "treatment stage_a_us")
            stage_b_us = _nonnegative_int(
                treatment_group.get("stage_b_us"), "treatment stage_b_us")
            prefill_us = _nonnegative_int(
                treatment_group.get("prefill_us"), "treatment prefill_us")
            decode_us = _nonnegative_int(
                treatment_group.get("decode_us"), "treatment decode_us")
            treatment_compute_us = prefill_us + decode_us
            treatment_route_us = stage_a_us + stage_b_us + treatment_tail_us
            if treatment_compute_us != treatment_route_us:
                raise AnalysisError("treatment route timing does not close")
            if treatment_route_us > treatment_wall_us:
                raise AnalysisError("treatment route time exceeds request_wall_us")
            treatment_residual_us = treatment_wall_us - treatment_route_us
            treatment_phone_us = stage_a_us + stage_b_us
            treatment_gap_us = treatment_phone_us + treatment_residual_us
            if treatment_gap_us <= 0:
                raise AnalysisError("treatment has no non-tail interval")
            rows.append({
                "pair_index": pair_index,
                "group_index": batch_index,
                "control_wall_us": control_wall_us,
                "treatment_wall_us": treatment_wall_us,
                "treatment_tail_us": treatment_tail_us,
                "treatment_phone_us": treatment_phone_us,
                "treatment_residual_us": treatment_residual_us,
                "treatment_gap_us": treatment_gap_us,
            })
    if len(set(pair_indexes)) != len(pair_indexes):
        raise AnalysisError("summary duplicates pair indexes")
    if sorted(pair_indexes) != list(range(len(pair_indexes))):
        raise AnalysisError("summary pair indexes are not contiguous")
    return rows


def analyze_row(row, target, gap_power_ratios):
    if not isinstance(target, Fraction) or target <= 0 or target >= 1:
        raise AnalysisError("target must be a fraction strictly between zero and one")
    control_wall = row["control_wall_us"]
    tail = row["treatment_tail_us"]
    gap = row["treatment_gap_us"]

    target_budget = target * control_wall
    same_tail_max_gap = (target_budget - tail) / gap
    sensitivity = []
    for gap_ratio in gap_power_ratios:
        if not isinstance(gap_ratio, Fraction) or gap_ratio < 0:
            raise AnalysisError("gap power ratios must be nonnegative fractions")
        max_tail_ratio = (target_budget - gap_ratio * gap) / tail
        sensitivity.append({
            "gap_power_ratio": _fraction_record(gap_ratio),
            "max_tail_power_ratio": _fraction_record(max_tail_ratio),
            "nonnegative_tail_power_feasible": max_tail_ratio >= 0,
        })

    return {
        **row,
        "target_energy_ratio": _fraction_record(target),
        "treatment_wall_vs_control": _fraction_record(
            Fraction(row["treatment_wall_us"], control_wall)),
        "tail_time_vs_control": _fraction_record(Fraction(tail, control_wall)),
        "gap_time_vs_control": _fraction_record(Fraction(gap, control_wall)),
        "max_gap_power_ratio_at_equal_tail_power": _fraction_record(
            same_tail_max_gap),
        "equal_tail_power_has_nonnegative_solution": same_tail_max_gap >= 0,
        "sensitivity": sensitivity,
    }


def analyze_summary(summary, target=Fraction(9, 10), gap_power_bps=None):
    if gap_power_bps is None:
        gap_power_bps = [0, 500, 1000, 1500, 2000]
    ratios = []
    for index, value in enumerate(gap_power_bps):
        if type(value) is not int or value < 0:
            raise AnalysisError(f"gap_power_bps[{index}] must be nonnegative")
        ratios.append(Fraction(value, 10_000))
    rows = [analyze_row(row, target, ratios) for row in extract_timing_rows(summary)]
    return {
        "schema": "s11-gpu-board-feasibility-v1",
        "scope": "MECHANICS_ONLY",
        "physical_energy_claim": "NONE",
        "model": (
            "E_treatment/E_control = tail_power_ratio*tail_time/control_time + "
            "gap_power_ratio*gap_time/control_time"),
        "rows": rows,
        "limitations": [
            "timing alone does not measure GPU-board power",
            "tail and gap power are sensitivity variables, not observations",
            "phone and total-system energy are excluded",
        ],
    }


def load_summary(path):
    try:
        resolved = pathlib.Path(path).resolve()
        data = resolved.read_bytes()
        text = data.decode("ascii")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot load summary: {exc}") from exc
    return value, {
        "path": str(resolved),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--target-num", type=int, default=9)
    parser.add_argument("--target-den", type=int, default=10)
    parser.add_argument(
        "--gap-power-bps", default="0,500,1000,1500,2000",
        help="comma-separated gap-power ratios in basis points of control power")
    args = parser.parse_args(argv)
    if args.target_den <= 0 or args.target_num <= 0 or args.target_num >= args.target_den:
        parser.error("target must satisfy 0 < target-num < target-den")
    try:
        gap_power_bps = [int(value) for value in args.gap_power_bps.split(",")]
    except ValueError as exc:
        parser.error(f"invalid --gap-power-bps: {exc}")
    summary, input_artifact = load_summary(args.summary)
    result = analyze_summary(
        summary,
        Fraction(args.target_num, args.target_den),
        gap_power_bps)
    result["input_artifact"] = input_artifact
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
