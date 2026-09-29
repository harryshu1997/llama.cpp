#!/usr/bin/env python3
"""Compare a matched phone-arbiter B-A-A-B physical screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


RUN_SCHEMA = "s42-fp16-burstgpt-gpu-wavefront-run-v1"
SCHEMA = "s42-fp16-burstgpt-phone-arbiter-abba-v1"
MINIMUM_FLEET_SAVING_PCT = 5.0


class AnalysisError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read {path}") from exc
    require(type(value) is dict, f"invalid object: {path}")
    return value


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def positive(value: object, name: str) -> float:
    require(
        type(value) in {int, float}
        and math.isfinite(value)
        and value > 0,
        f"invalid {name}",
    )
    return float(value)


def marked_result(path: Path, marker: str) -> dict[str, Any]:
    rows = [
        line.partition(marker)[2]
        for line in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if marker in line
    ]
    require(len(rows) == 1, f"non-unique {marker.strip()} receipt")
    value = json.loads(rows[0])
    require(type(value) is dict, f"invalid {marker.strip()} receipt")
    return value


def validate_run(path: Path, phone_arbiter: bool) -> dict[str, Any]:
    result = read_object(path)
    unsigned = dict(result)
    claimed_hash = unsigned.pop("record_sha256", None)
    require(
        result.get("schema") == RUN_SCHEMA
        and result.get("status") == "PASS"
        and result.get("mode") == "control"
        and result.get("phone_arbiter") is phone_arbiter
        and result.get("energy_claim_eligible") is False
        and type(claimed_hash) is str
        and hashlib.sha256(canonical(unsigned)).hexdigest() == claimed_hash,
        f"run identity: {path}",
    )
    capture = path.parent
    require(capture.name.endswith(".wavefront"), f"capture root: {path}")
    root = Path(str(capture).removesuffix(".wavefront"))
    phone_capture = Path(str(root) + ".phone-capture")
    qwen = read_object(root / "RESULT.json")
    gemma = read_object(capture / "GEMMA_RESULT.json")
    router = marked_result(
        phone_capture / "router.log",
        "RESIDENTROUTER ",
    )
    require(
        qwen.get("status") == "PASS"
        and gemma.get("status") == "PASS"
        and router.get("status") == "ok",
        f"component status: {path}",
    )
    qwen_rows = qwen.get("request_results")
    gemma_rows = gemma.get("request_results")
    require(
        type(qwen_rows) is list
        and type(gemma_rows) is list
        and len(qwen_rows) == 3
        and len(gemma_rows) == 1,
        f"request count: {path}",
    )
    qwen_work = tuple(
        (
            row.get("request_index"),
            row.get("input_tokens"),
            row.get("output_tokens"),
            tuple(row.get("tokens", [])),
        )
        for row in qwen_rows
    )
    gemma_work = tuple(
        (
            row.get("request_index"),
            row.get("input_tokens"),
            row.get("output_tokens"),
            tuple(row.get("token_ids", [])),
        )
        for row in gemma_rows
    )
    require(
        all(len(row[3]) == row[2] for row in qwen_work + gemma_work),
        f"exact output lengths: {path}",
    )
    paid_start_ns = qwen.get("paid_start_ns")
    qwen_end_ns = qwen.get("qwen_end_ns")
    require(
        type(paid_start_ns) is int
        and type(qwen_end_ns) is int
        and paid_start_ns < qwen_end_ns,
        f"Qwen interval: {path}",
    )
    energy = result.get("energy")
    require(type(energy) is dict, f"energy: {path}")
    metrics = {
        "cpu_package_j": positive(energy.get("cpu_package_j"), "CPU energy"),
        "duration_s": positive(result.get("paid_duration_s"), "duration"),
        "fleet_j": positive(energy.get("fleet_j"), "fleet energy"),
        "gemma_decode_s": positive(gemma_rows[0].get("decode_us"), "Gemma decode") / 1e6,
        "gemma_prefill_s": positive(gemma_rows[0].get("prefill_us"), "Gemma prefill") / 1e6,
        "gemma_wall_s": positive(gemma_rows[0].get("route_wall_us"), "Gemma wall") / 1e6,
        "gpu_board_j": positive(energy.get("gpu_board_j"), "GPU energy"),
        "phone_j": positive(energy.get("phone_j"), "phone energy"),
        "qwen_s": (qwen_end_ns - paid_start_ns) / 1e9,
        "server_j": positive(energy.get("server_j"), "server energy"),
    }
    require(
        math.isclose(
            metrics["fleet_j"],
            metrics["server_j"] + metrics["phone_j"],
        ),
        f"fleet energy sum: {path}",
    )
    bridge = result.get("bridge")
    require(type(bridge) is dict, f"bridge receipt: {path}")
    if phone_arbiter:
        require(
            result.get("admission") == "PHONE_ARBITER_MECHANICS_ONLY"
            and gemma.get("ffn_route", {}).get("decode_columns") == 6144
            and bridge.get("protected_done_observed") is True
            and bridge.get("filler_calls")
            == (gemma_work[0][2] - 1) * 23
            and bridge.get("filler_upper_violations") == 0
            and bridge.get("guard_violations") == 0
            and bridge.get("idle_lower_violations") == 0
            and bridge.get("protected_pending_after_filler") == 0
            and router.get("terminate_requested") is True
            and router.get("requests")
            == bridge.get("protected_calls") + bridge.get("filler_calls"),
            f"phone arbiter mechanics: {path}",
        )
    else:
        require(
            result.get("admission") == "CONTROL_CALIBRATION_ONLY"
            and gemma.get("ffn_route") is None
            and router.get("requests") == bridge.get("calls"),
            f"control phone route: {path}",
        )
    return {
        "artifact_sha256": digest(path),
        "metrics": metrics,
        "paid_start_ns": paid_start_ns,
        "profile_sha256": result.get("artifacts", {}).get("profile_sha256"),
        "tokens": {
            "gemma": gemma_work[0][3],
            "qwen": tuple(row[3] for row in qwen_work),
        },
        "work_shape": {
            "gemma": tuple(row[:3] for row in gemma_work),
            "qwen": tuple(row[:3] for row in qwen_work),
        },
    }


def mean(rows: list[dict[str, Any]], metric: str) -> float:
    return sum(row["metrics"][metric] for row in rows) / len(rows)


def saving_pct(control: float, treatment: float) -> float:
    return 100.0 * (1.0 - treatment / control)


def analyze(paths: dict[str, Path]) -> dict[str, Any]:
    run_order = (
        "treatment_r1",
        "control_r1",
        "control_r2",
        "treatment_r2",
    )
    runs = {
        name: validate_run(path, name.startswith("treatment"))
        for name, path in paths.items()
    }
    require(set(runs) == set(run_order), "B-A-A-B input set")
    starts = [runs[name]["paid_start_ns"] for name in run_order]
    require(starts == sorted(starts) and len(set(starts)) == 4, "physical B-A-A-B order")
    work = [runs[name]["work_shape"] for name in run_order]
    require(all(value == work[0] for value in work[1:]), "equal work shape")
    qwen_tokens = [runs[name]["tokens"]["qwen"] for name in run_order]
    require(
        all(value == qwen_tokens[0] for value in qwen_tokens[1:]),
        "Qwen exact output stability",
    )
    control_gemma = [row["tokens"]["gemma"] for row in (
        runs["control_r1"], runs["control_r2"]
    )]
    treatment_gemma = [row["tokens"]["gemma"] for row in (
        runs["treatment_r1"], runs["treatment_r2"]
    )]
    require(
        control_gemma[0] == control_gemma[1]
        and treatment_gemma[0] == treatment_gemma[1],
        "Gemma within-arm output stability",
    )
    matches = sum(
        left == right
        for left, right in zip(
            control_gemma[0], treatment_gemma[0], strict=True
        )
    )
    common_prefix = 0
    for left, right in zip(
        control_gemma[0], treatment_gemma[0], strict=True
    ):
        if left != right:
            break
        common_prefix += 1
    gemma_quality = {
        "common_prefix_tokens": common_prefix,
        "exact": control_gemma[0] == treatment_gemma[0],
        "positional_agreement_pct": (
            100.0 * matches / len(control_gemma[0])
        ),
        "positional_matches": matches,
        "tokens": len(control_gemma[0]),
    }
    profiles = [runs[name]["profile_sha256"] for name in run_order]
    require(len(set(profiles)) == 1 and profiles[0], "matched profile")

    controls = [runs["control_r1"], runs["control_r2"]]
    treatments = [runs["treatment_r1"], runs["treatment_r2"]]
    metric_names = tuple(controls[0]["metrics"])
    control_mean = {
        name: mean(controls, name) for name in metric_names
    }
    treatment_mean = {
        name: mean(treatments, name) for name in metric_names
    }
    savings = {
        name + "_pct": saving_pct(
            control_mean[name], treatment_mean[name]
        )
        for name in metric_names
    }
    pair_fleet_savings = [
        saving_pct(
            controls[index]["metrics"]["fleet_j"],
            treatments[index]["metrics"]["fleet_j"],
        )
        for index in range(2)
    ]
    pair_duration_savings = [
        saving_pct(
            controls[index]["metrics"]["duration_s"],
            treatments[index]["metrics"]["duration_s"],
        )
        for index in range(2)
    ]
    gates = {
        "each_pair_duration_lower": all(
            value > 0 for value in pair_duration_savings
        ),
        "each_pair_fleet_energy_lower": all(
            value > 0 for value in pair_fleet_savings
        ),
        "gemma_approximate_quality_at_least_90pct": (
            gemma_quality["positional_agreement_pct"] >= 90.0
        ),
        "gemma_within_arm_output_stable": True,
        "gemma_decode_mean_lower": savings["gemma_decode_s_pct"] > 0,
        "mean_fleet_saving_at_least_5pct": (
            savings["fleet_j_pct"] >= MINIMUM_FLEET_SAVING_PCT
        ),
        "phone_mechanics_pass": True,
        "qwen_exact_output_stable": True,
        "work_shape_equal": True,
    }
    if all(gates.values()):
        admission = "DIRECTION_PASS_THIRD_PAIR_REQUIRED"
    elif (
        gates["each_pair_duration_lower"]
        and gates["each_pair_fleet_energy_lower"]
        and not gates["gemma_decode_mean_lower"]
    ):
        admission = "APPARENT_SAVING_NOT_ATTRIBUTED_TO_PHONE"
    else:
        admission = "RETAIN_CONTROL_PHONE_ROUTE"
    output: dict[str, Any] = {
        "admission": admission,
        "control_mean": control_mean,
        "energy_claim_eligible": False,
        "gates": gates,
        "minimum_fleet_saving_pct": MINIMUM_FLEET_SAVING_PCT,
        "pair_duration_savings_pct": pair_duration_savings,
        "pair_fleet_savings_pct": pair_fleet_savings,
        "run_order": list(run_order),
        "quality": {"gemma": gemma_quality, "qwen_exact": True},
        "runs": {
            name: {
                "artifact_sha256": runs[name]["artifact_sha256"],
                "metrics": runs[name]["metrics"],
            }
            for name in run_order
        },
        "savings_pct": savings,
        "schema": SCHEMA,
        "status": "PASS",
        "treatment_mean": treatment_mean,
    }
    output["record_sha256"] = hashlib.sha256(canonical(output)).hexdigest()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment-r1", type=Path, required=True)
    parser.add_argument("--control-r1", type=Path, required=True)
    parser.add_argument("--control-r2", type=Path, required=True)
    parser.add_argument("--treatment-r2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.output.is_absolute()
        or args.output.exists()
        or not args.output.parent.is_dir()
    ):
        parser.error("output must be an unused absolute path")
    paths = {
        name: getattr(args, name)
        for name in (
            "treatment_r1",
            "control_r1",
            "control_r2",
            "treatment_r2",
        )
    }
    try:
        result = analyze(paths)
    except (AnalysisError, OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"phone arbiter A-B-B-A analysis failed: {exc}\n")
    args.output.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "admission": result["admission"],
        "duration_saving_pct": result["savings_pct"]["duration_s_pct"],
        "fleet_saving_pct": result["savings_pct"]["fleet_j_pct"],
        "status": result["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
