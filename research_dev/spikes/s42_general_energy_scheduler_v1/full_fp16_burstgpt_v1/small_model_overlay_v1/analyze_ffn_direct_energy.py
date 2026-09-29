#!/usr/bin/env python3
"""Fit a signed direct FFN route-energy record from isolated ABBA data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
S42_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[4]
MIXED_ROOT = S42_ROOT / "mixed_model_trace_v1"
KERNEL_ENERGY = S42_ROOT / "kernel_energy_v1"
WHOLE_TASK = S42_ROOT / "whole_task_phone_v1"
sys.path[:0] = [
    str(REPO_ROOT),
    str(S42_ROOT),
    str(MIXED_ROOT),
    str(KERNEL_ENERGY),
    str(WHOLE_TASK),
    str(HERE),
]

import analyze_resident_route_campaign as resident  # noqa: E402
import energy_common  # noqa: E402
import materialize_ffn_split_route as materializer  # noqa: E402
import run_fp16_small_overlay as overlay  # noqa: E402


CAMPAIGN_SCHEMA = "s42-llama1b-ffn-direct-energy-campaign-v1"
DIRECT_SCHEMA = "s42-llama1b-ffn-direct-energy-v1"
ANALYSIS_SCHEMA = "s42-llama1b-ffn-direct-energy-analysis-v1"
CPU_ROUTE = "desktop-cpu"
SPLIT_ROUTE = "cpu-phone-ffn-split"
BOUNDARY = (
    "incremental-cpu-package-gpu-board-whole-phone-above-resident-idle-v1"
)
METHOD = "isolated-sequential-incremental-above-resident-idle-v1"
QUALIFICATION_PAD_PPM = 100_000
T95 = {
    2: 12.706204736,
    3: 4.30265273,
    4: 3.182446305,
    5: 2.776445105,
    6: 2.570581836,
    7: 2.446911851,
    8: 2.364624252,
    9: 2.306004135,
    10: 2.262157163,
}


class AnalysisError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"object expected: {path}")
    return value


def read_resource_samples(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, raw in enumerate(path.read_bytes().splitlines(keepends=True), 1):
        require(raw.endswith(b"\n"), f"resource line framing: {line_number}")
        row = json.loads(raw)
        require(
            type(row) is dict
            and row.get("schema") == "s42-six-model-resource-sample-v1"
            and overlay.physical.run_trace.canonical(row) == raw,
            f"resource sample identity: {line_number}",
        )
        rows.append(row)
    require(len(rows) >= 3, "resource sample count")
    return rows


def read_anchor(path: Path) -> dict[str, Any]:
    value = load(path)
    require(
        value.get("schema") == "s41-phone-clock-anchor-v1"
        and type(value.get("boot_id")) is str
        and type(value.get("host_midpoint_ns")) is int
        and type(value.get("phone_uptime_ns")) is int
        and type(value.get("round_trip_ns")) is int
        and value["round_trip_ns"] <= 500_000_000,
        f"clock anchor: {path}",
    )
    return value


class PhoneClock:
    def __init__(self, before: dict[str, Any], after: dict[str, Any]) -> None:
        require(before["boot_id"] == after["boot_id"], "phone reboot")
        host_delta = after["host_midpoint_ns"] - before["host_midpoint_ns"]
        phone_delta = after["phone_uptime_ns"] - before["phone_uptime_ns"]
        require(host_delta > 0 and phone_delta > 0, "clock direction")
        self.slope = host_delta / phone_delta
        require(abs(self.slope - 1.0) <= 0.002, "phone clock drift")
        self.host_origin = before["host_midpoint_ns"]
        self.phone_origin = before["phone_uptime_ns"]

    def phone_ns(self, host_ns: int) -> int:
        return self.phone_origin + round(
            (host_ns - self.host_origin) / self.slope
        )


def paired_ratio_ci95(values: list[float]) -> dict[str, float | int]:
    require(len(values) >= 2, "paired ratio sample count")
    value_mean = mean(values)
    critical = T95.get(len(values), 1.96 if len(values) >= 30 else 2.228)
    half_width = critical * stdev(values) / math.sqrt(len(values))
    return {
        "ci95_high": value_mean + half_width,
        "ci95_low": value_mean - half_width,
        "mean": value_mean,
        "samples": len(values),
    }


def predict(coefficients: dict[str, int], row: dict[str, Any]) -> int:
    return max(
        1,
        coefficients["fixed"]
        + coefficients["input_token"] * row["input_tokens"]
        + coefficients["output_token"] * row["output_tokens"],
    )


def local_idle_power(
    idle_by_batch: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    result = {}
    for batch_id, rows in idle_by_batch.items():
        require(
            len(rows) == 2 and {row["position"] for row in rows} == {"before", "after"},
            f"bracketing idle samples: {batch_id}",
        )
        result[batch_id] = {
            name: mean(row[name] / row["duration_s"] for row in rows)
            for name in (
                "cpu_package_energy_j",
                "gpu_board_energy_j",
                "whole_phone_energy_j",
            )
        }
        result[batch_id]["accounted_fleet_w"] = sum(
            result[batch_id].values()
        )
    return result


def add_energy(
    campaign: dict[str, Any],
    resources: list[dict[str, Any]],
    phone_rows: list[dict[str, float | int]],
    clock: PhoneClock,
) -> list[dict[str, Any]]:
    measured = []
    for source in campaign["cases"]:
        row = json.loads(json.dumps(source))
        start_ns = row["paid_start_monotonic_ns"]
        end_ns = row["paid_end_monotonic_ns"]
        server = overlay.physical.run_trace.server_energy_summary(
            resources, start_ns, end_ns
        )
        phone = energy_common.phone_energy_summary(
            phone_rows,
            clock.phone_ns(start_ns),
            clock.phone_ns(end_ns),
        )
        require(
            abs(server["cpu_package_energy_j"])
            < 1_000_000
            and abs(server["gpu_board_energy_j"]) < 1_000_000
            and abs(phone["whole_phone_energy_j"]) < 1_000_000,
            f"energy bounds: {row['case_id']}",
        )
        row.update({
            "accounted_fleet_energy_j": (
                server["server_compute_device_energy_j"]
                + phone["whole_phone_energy_j"]
            ),
            "cpu_package_energy_j": server["cpu_package_energy_j"],
            "gpu_board_energy_j": server["gpu_board_energy_j"],
            "whole_phone_energy_j": phone["whole_phone_energy_j"],
        })
        measured.append(row)
    idle_by_batch: dict[str, list[dict[str, Any]]] = {}
    for row in measured:
        if row["route"] == "idle":
            idle_by_batch.setdefault(row["batch_id"], []).append(row)
    baselines = local_idle_power(idle_by_batch)
    for row in measured:
        if row["route"] == "idle":
            continue
        baseline = baselines[row["batch_id"]]
        raw_j = (
            row["accounted_fleet_energy_j"]
            - baseline["accounted_fleet_w"] * row["duration_s"]
        )
        row["idle_baseline_power_w"] = baseline["accounted_fleet_w"]
        row["marginal_energy_above_resident_idle_j"] = raw_j
        row["marginal_energy_uj"] = max(1, round(raw_j * 1_000_000))
    return measured


def route_fit(
    rows: list[dict[str, Any]], train_max_cycle: int
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    training = [row for row in rows if row["cycle"] <= train_max_cycle]
    heldout = [row for row in rows if row["cycle"] > train_max_cycle]
    require(training and heldout, "train and heldout route observations")
    fit_rows = [
        {**row, "mixed_request_index": row["overlay_request_index"]}
        for row in training
    ]
    fitted = resident.fit_nonnegative(
        fit_rows, lambda row: float(row["marginal_energy_uj"])
    )
    upper_error_ppm = min(
        1_000_000,
        fitted["upper_error_ppm"] + QUALIFICATION_PAD_PPM,
    )
    violations = []
    for row in heldout:
        prediction = predict(fitted["coefficients"], row)
        upper = math.ceil(prediction * (1 + upper_error_ppm / 1_000_000))
        violations.append({
            "actual_uj": row["marginal_energy_uj"],
            "case_id": row["case_id"],
            "prediction_uj": prediction,
            "upper_uj": upper,
            "violation": row["marginal_energy_uj"] > upper,
        })
    return {
        "boundary_id": BOUNDARY,
        "cost_uj": {
            "fixed": fitted["coefficients"]["fixed"],
            "input_token": fitted["coefficients"]["input_token"],
            "kind": "affine_tokens_v1",
            "output_token": fitted["coefficients"]["output_token"],
        },
        "lower_error_ppm": fitted["lower_error_ppm"],
        "status": "measured",
        "upper_error_ppm": upper_error_ppm,
    }, violations, sum(row["violation"] for row in violations)


def paired_ratios(
    rows: list[dict[str, Any]],
) -> tuple[list[float], list[dict[str, Any]]]:
    pairs = []
    ratios = []
    pair_ids = sorted({row["pair_id"] for row in rows})
    for pair_id in pair_ids:
        selected = [row for row in rows if row["pair_id"] == pair_id]
        by_route = {
            route: [row for row in selected if row["route"] == route]
            for route in (CPU_ROUTE, SPLIT_ROUTE)
        }
        require(
            all(by_route.values())
            and {
                row["overlay_request_index"] for row in by_route[CPU_ROUTE]
            }
            == {
                row["overlay_request_index"] for row in by_route[SPLIT_ROUTE]
            },
            f"matched pair work: {pair_id}",
        )
        cpu_uj = sum(row["marginal_energy_uj"] for row in by_route[CPU_ROUTE])
        split_uj = sum(row["marginal_energy_uj"] for row in by_route[SPLIT_ROUTE])
        require(cpu_uj > 0 and split_uj > 0, f"positive pair energy: {pair_id}")
        ratio = split_uj / cpu_uj
        ratios.append(ratio)
        pairs.append({
            "cpu_marginal_energy_uj": cpu_uj,
            "pair_id": pair_id,
            "ratio": ratio,
            "split_marginal_energy_uj": split_uj,
        })
    return ratios, pairs


def analyze(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign = load(args.result)
    manifest = materializer.verified_record(
        args.ffn_manifest, materializer.FFN_MANIFEST_SCHEMA
    )
    policy = materializer.verified_record(
        args.ffn_policy, materializer.FFN_POLICY_SCHEMA
    )
    require(
        campaign.get("schema") == CAMPAIGN_SCHEMA
        and campaign.get("status") == "PASS"
        and campaign.get("input_sha256", {}).get("ffn_manifest")
        == digest(args.ffn_manifest)
        and campaign.get("input_sha256", {}).get("ffn_policy")
        == digest(args.ffn_policy),
        "campaign identity",
    )
    require(
        policy.get("evidence", {}).get("manifest_record_sha256")
        == manifest["record_sha256"],
        "manifest and policy binding",
    )
    resources = read_resource_samples(args.resource_samples)
    require(
        campaign.get("resource_samples", {}).get("sha256")
        == digest(args.resource_samples),
        "resource sample binding",
    )
    phone_rows = energy_common.read_phone_samples(args.phone_samples)
    before = read_anchor(args.clock_before)
    after = read_anchor(args.clock_after)
    clock = PhoneClock(before, after)
    measured = add_energy(campaign, resources, phone_rows, clock)
    request_rows = [row for row in measured if row["route"] != "idle"]
    idle_rows = [row for row in measured if row["route"] == "idle"]
    windows = sorted(
        (row["paid_start_monotonic_ns"], row["paid_end_monotonic_ns"])
        for row in request_rows
    )
    non_overlapping = all(
        left[1] <= right[0] for left, right in zip(windows, windows[1:])
    )
    require(non_overlapping, "non-overlapping request windows")
    positive_margins = all(
        row["marginal_energy_above_resident_idle_j"] > 0
        for row in request_rows
    )
    ratios, pair_rows = paired_ratios(request_rows)
    ratio_ci = paired_ratio_ci95(ratios)
    max_cycle = max(row["cycle"] for row in request_rows)
    require(max_cycle >= 2, "heldout cycle")
    split_rows = [row for row in request_rows if row["route"] == SPLIT_ROUTE]
    cpu_rows = [row for row in request_rows if row["route"] == CPU_ROUTE]
    split_energy, heldout, violations = route_fit(
        split_rows, max_cycle - 1
    )
    cpu_energy, cpu_heldout, cpu_violations = route_fit(
        cpu_rows, max_cycle - 1
    )
    materializer.validate_split_contract(
        args.split_log, manifest, policy
    )
    split_log = materializer.parse_split_log(args.split_log)
    max_tokens = manifest["split_contract"]["max_tokens"]
    phone_buckets = [
        bucket
        for bucket in policy["compiled_buckets"]
        if bucket["phone_columns"] > 0
    ]
    expected_prewarm = max(
        phone_buckets, key=lambda bucket: bucket["max_tokens"]
    )
    prewarm = campaign.get("split_accelerator_prewarm", {})
    prewarm_physical = (
        prewarm.get("status") == "PASS"
        and prewarm.get("shape") == {
            "input_tokens": expected_prewarm["max_tokens"],
            "phone_columns": expected_prewarm["phone_columns"],
        }
        and type(prewarm.get("duration_s")) in (int, float)
        and prewarm["duration_s"] > 0
        and prewarm.get("response", {}).get("runtime_prompt_tokens")
            == expected_prewarm["max_tokens"]
    )
    split_requests_with_phone_work = sum(
        materializer.phone_columns_for_tokens(
            policy, min(row["input_tokens"], max_tokens)
        )
        > 0
        for row in split_rows
    )
    minimum_calls = (
        split_requests_with_phone_work
        * len(manifest["geometry"]["resident_layer_ids"])
    )
    prewarm_calls = len(manifest["geometry"]["resident_layer_ids"])
    physical_calls = (
        split_requests_with_phone_work > 0
        and split_log["summary"]["calls"]
            >= minimum_calls + prewarm_calls
    )
    gates = {
        "accelerator_shape_prewarm": prewarm_physical,
        "direct_split_energy_ci_margin": ratio_ci["ci95_high"] < 0.95,
        "heldout_split_upper_bound": violations == 0,
        "positive_request_marginal_energy": positive_margins,
        "request_windows_non_overlapping": non_overlapping,
        "split_calls_physically_observed": physical_calls,
    }
    qualified = all(gates.values())
    qualification = {
        "abba_cycle_count": campaign["abba_cycles"],
        "gates": gates,
        "heldout_upper_bound_violations": violations,
        "idle_baseline_sample_count": len(idle_rows),
        "idle_baseline_subtracted": True,
        "max_concurrent_measured_requests": 1,
        "method": METHOD,
        "repeat_count": len(ratios),
        "request_windows_non_overlapping": non_overlapping,
        "split_over_cpu_ratio_ci95": ratio_ci,
        "status": "PASS" if qualified else "FAIL",
    }
    record = {
        "binding": {
            "manifest_record_sha256": manifest["record_sha256"],
            "model_sha256": manifest["model"]["sha256"],
            "policy_execution_sha256": (
                materializer.policy_execution_sha256(policy)
            ),
            "policy_record_sha256": policy["record_sha256"],
            "route_id": SPLIT_ROUTE,
        },
        "evidence": {
            "clock_after_sha256": digest(args.clock_after),
            "clock_before_sha256": digest(args.clock_before),
            "measurement_result_sha256": digest(args.result),
            "phone_samples_sha256": digest(args.phone_samples),
            "resource_samples_sha256": digest(args.resource_samples),
            "split_log_sha256": digest(args.split_log),
        },
        "qualification": qualification,
        "route_energy": split_energy,
        "schema": DIRECT_SCHEMA,
        "status": "PASS" if qualified else "FAIL",
    }
    record["record_sha256"] = hashlib.sha256(canonical(record)).hexdigest()
    analysis = {
        "clock_slope_host_per_phone": clock.slope,
        "cpu_route_energy_diagnostic": cpu_energy,
        "cpu_route_heldout": cpu_heldout,
        "cpu_route_heldout_violations": cpu_violations,
        "direct_energy_record_sha256": record["record_sha256"],
        "idle_samples": idle_rows,
        "input_sha256": record["evidence"],
        "paired_batches": pair_rows,
        "qualification": qualification,
        "request_measurements": request_rows,
        "schema": ANALYSIS_SCHEMA,
        "split_call_summary": split_log,
        "split_requests_with_phone_work": split_requests_with_phone_work,
        "split_route_heldout": heldout,
        "status": record["status"],
    }
    return record, analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--resource-samples", type=Path, required=True)
    parser.add_argument("--phone-samples", type=Path, required=True)
    parser.add_argument("--clock-before", type=Path, required=True)
    parser.add_argument("--clock-after", type=Path, required=True)
    parser.add_argument("--ffn-manifest", type=Path, required=True)
    parser.add_argument("--ffn-policy", type=Path, required=True)
    parser.add_argument("--split-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (
        args.result,
        args.resource_samples,
        args.phone_samples,
        args.clock_before,
        args.clock_after,
        args.ffn_manifest,
        args.ffn_policy,
        args.split_log,
    ):
        require(path.is_file(), f"missing input: {path}")
    require(
        args.output.is_absolute()
        and args.analysis_output.is_absolute()
        and not args.output.exists()
        and not args.analysis_output.exists(),
        "new absolute outputs",
    )
    record, analysis = analyze(args)
    args.output.write_bytes(canonical(record))
    args.analysis_output.write_bytes(canonical(analysis))
    print(json.dumps({
        "output": str(args.output),
        "ratio_ci95": record["qualification"]["split_over_cpu_ratio_ci95"],
        "status": record["status"],
    }, sort_keys=True))
    return 0 if record["status"] == "PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AnalysisError, energy_common.EnergyError) as error:
        print(f"S42_FFN_DIRECT_ANALYSIS_ERROR: {error}")
        raise SystemExit(2)
