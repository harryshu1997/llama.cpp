#!/usr/bin/env python3
"""Validate and compare one matched GPU-overflow control/treatment pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import load_execution_plan  # noqa: E402


MODEL_SHA256 = (
    "ed76f2183d2d1d65091986033023e6c7"
    "8d27f6276c1b0c5826cc92acf73538cf"
)
TRACE_SHA256 = (
    "b20a9ba66ee3558d835a0e19ed3cfa4"
    "c31a4a9e8b4f9c085b29a14f80250a0ff"
)


class AnalysisError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def validate_phone_work(plan, split: dict[str, Any]) -> None:
    bridge = split.get("bridge")
    ffn = split.get("ffn")
    shapes = split.get("shapes")
    require(
        plan.offload is not None
        and type(bridge) is dict
        and type(ffn) is dict
        and type(shapes) is list,
        "treatment phone summaries",
    )
    policy = plan.offload.split
    layer_count = len(policy.layer_ids)
    shape_calls: list[tuple[int, int]] = []
    for row in shapes:
        tokens = row.get("tokens") if type(row) is dict else None
        calls = row.get("calls") if type(row) is dict else None
        columns = row.get("columns") if type(row) is dict else None
        require(
            type(tokens) is int
            and type(calls) is int
            and calls > 0
            and calls % layer_count == 0
            and columns == policy.columns_for(tokens)
            and columns > 0,
            "treatment phone shape accounting",
        )
        shape_calls.append((tokens, calls))
    work = policy.summarize(shape_calls)
    expected = plan.expected_work
    require(
        shape_calls
        and work.token_rows % layer_count == 0
        and bridge.get("status") == "ok"
        and ffn.get("status") == "ok"
        and bridge.get("allocator") == plan.offload.transport.allocator
        and bridge.get("max_wire_bytes")
            == plan.offload.transport.max_wire_bytes
        and bridge.get("calls") == work.calls
        and ffn.get("calls") == work.calls
        and bridge.get("upload_bytes") == work.upload_bytes
        and bridge.get("download_bytes") == work.download_bytes
        and ffn.get("upload_bytes") == work.upload_bytes
        and ffn.get("download_bytes") == work.download_bytes
        and bridge.get("reset_recoveries") == 0
        and expected["phone_calls_min"] <= work.calls
            <= expected["phone_calls_max"]
        and expected["phone_transfer_bytes_min"] <= work.upload_bytes
            <= expected["phone_transfer_bytes_max"]
        and expected["phone_macs_min"] <= work.phone_macs
            <= expected["phone_macs_max"]
        and expected["phone_token_rows_per_layer_min"]
            <= work.token_rows // layer_count
            <= expected["phone_token_rows_per_layer_max"],
        "treatment phone work",
    )


def validate_arm(
    arm: str,
    result_path: Path,
    phone_path: Path,
    plan_path: Path,
) -> dict[str, Any]:
    expected_mode = "cuda-cpu" if arm == "control" else "cuda-cpu-op15"
    result = read_object(result_path)
    phone = read_object(phone_path)
    plan = load_execution_plan(plan_path)
    require(
        result.get("schema") == "s41-gemma-gpu-trace-v2"
        and result.get("status") == "PASS"
        and result.get("mode") == expected_mode,
        f"{arm} result identity",
    )
    require(
        phone.get("schema") == "s41-phone-energy-v3"
        and phone.get("status") == "PASS"
        and phone.get("boundary") == "paid_trace_interval",
        f"{arm} phone energy identity",
    )
    require(
        plan.execution_mode == expected_mode
        and result.get("scheduler_plan_sha256") == plan.plan_sha256
        and result.get("preflight", {}).get("scheduler", {}).get("plan_sha256")
            == plan.plan_sha256,
        f"{arm} unified plan binding",
    )
    placement = plan.layer_placement
    require(
        placement is not None
        and placement.cpu_layer_spec == "0-22"
        and placement.gpu_layer_spec == "23-47"
        and placement.selected.runtime_gpu_layers == 25
        and int(result.get("n_gpu_layers")) == 25,
        f"{arm} layer placement",
    )
    require(
        plan.trace_sha256 == "sha256:" + TRACE_SHA256
        and plan.model_hashes == {"cold": "sha256:" + MODEL_SHA256}
        and result.get("model", {}).get("sha256") == MODEL_SHA256,
        f"{arm} workload epoch",
    )
    rows = result.get("request_results")
    metrics = result.get("metrics")
    require(
        type(rows) is list
        and len(rows) == 17
        and type(metrics) is dict
        and metrics.get("completed") == 17
        and metrics.get("output_tokens") == 6_919
        and sum(row.get("input_tokens", 0) for row in rows) == 11_476
        and sum(row.get("output_tokens", 0) for row in rows) == 6_919,
        f"{arm} completed work",
    )
    resources = result.get("resources")
    require(
        type(resources) is dict
        and resources.get("gpu_memory_total_bytes") == 17_175_674_880
        and resources.get("process_swap_max_bytes") == 0,
        f"{arm} memory gate",
    )
    server = result.get("server_energy")
    require(
        type(server) is dict
        and server.get("boundary") == "paid_trace_interval"
        and server.get("server_compute_device_energy_j", 0) > 0
        and server.get("cpu_package_energy_j", 0) > 0
        and server.get("gpu_board_energy_j", 0) > 0,
        f"{arm} server energy",
    )
    duration_s = (result["paid_end_ns"] - result["paid_start_ns"]) / 1e9
    require(
        abs(duration_s - metrics["makespan_s"]) <= 0.001
        and abs(duration_s - phone.get("duration_s", -1)) <= 0.001
        and phone.get("whole_phone_energy_j", 0) > 0,
        f"{arm} paid interval",
    )
    split = result.get("phone")
    if arm == "control":
        require(
            plan.offload is None
            and split == {"bridge": None, "ffn": None, "shapes": []},
            "control phone work",
        )
    else:
        require(type(split) is dict, "treatment phone work")
        validate_phone_work(plan, split)
    return {
        "cpu_package_j": server["cpu_package_energy_j"],
        "duration_s": duration_s,
        "fleet_j": (
            server["server_compute_device_energy_j"]
            + phone["whole_phone_energy_j"]
        ),
        "gpu_board_j": server["gpu_board_energy_j"],
        "phone_j": phone["whole_phone_energy_j"],
        "plan": plan,
        "result": result,
        "server_j": server["server_compute_device_energy_j"],
    }


def change_pct(control: float, treatment: float) -> float:
    return (treatment / control - 1.0) * 100.0


def analyze(
    control_result: Path,
    control_phone: Path,
    control_plan: Path,
    treatment_result: Path,
    treatment_phone: Path,
    treatment_plan: Path,
) -> dict[str, Any]:
    control = validate_arm(
        "control", control_result, control_phone, control_plan
    )
    treatment = validate_arm(
        "treatment", treatment_result, treatment_phone, treatment_plan
    )
    require(
        control["plan"].layer_placement.selected
        == treatment["plan"].layer_placement.selected,
        "arms use different GPU/CPU placement",
    )
    require(
        control["result"]["repeat_index"]
        == treatment["result"]["repeat_index"],
        "pair repeat index",
    )
    fleet_change = change_pct(control["fleet_j"], treatment["fleet_j"])
    server_change = change_pct(control["server_j"], treatment["server_j"])
    makespan_change = change_pct(control["duration_s"], treatment["duration_s"])
    saving = fleet_change < 0
    output: dict[str, Any] = {
        "boundary": "cpu-package+gpu-board+whole-phone-paid-cold17-interval-v1",
        "comparison": {
            "control": {
                key: control[key]
                for key in (
                    "cpu_package_j",
                    "duration_s",
                    "fleet_j",
                    "gpu_board_j",
                    "phone_j",
                    "server_j",
                )
            },
            "fleet_energy_change_pct": fleet_change,
            "makespan_change_pct": makespan_change,
            "server_energy_change_pct": server_change,
            "treatment": {
                key: treatment[key]
                for key in (
                    "cpu_package_j",
                    "duration_s",
                    "fleet_j",
                    "gpu_board_j",
                    "phone_j",
                    "server_j",
                )
            },
        },
        "energy_verdict": "SAVING" if saving else "REGRESSION",
        "gates": {
            "equal_completed_work": True,
            "fleet_energy_saving": saving,
            "same_gpu_cpu_layer_placement": True,
            "synchronized_whole_phone_energy": True,
            "treatment_zero_reset_recoveries": True,
            "unified_scheduler_plans_bound": True,
            "zero_swap": True,
        },
        "input_sha256": {
            "control_phone": sha256(control_phone),
            "control_plan": sha256(control_plan),
            "control_result": sha256(control_result),
            "treatment_phone": sha256(treatment_phone),
            "treatment_plan": sha256(treatment_plan),
            "treatment_result": sha256(treatment_result),
        },
        "placement": {
            "cpu_layer_spec": "0-22",
            "gpu_layer_spec": "23-47",
            "runtime_gpu_layers": 25,
        },
        "repeat_index": control["result"]["repeat_index"],
        "scheduler": {
            "control": {
                "plan_sha256": control["plan"].plan_sha256,
                "reason": control["plan"].decision.reason,
                "route_id": control["plan"].decision.route_id,
            },
            "treatment": {
                "plan_sha256": treatment["plan"].plan_sha256,
                "reason": treatment["plan"].decision.reason,
                "route_id": treatment["plan"].decision.route_id,
            },
        },
        "schema": "s42-gpu-overflow-physical-pair-v1",
        "status": "PASS",
        "trace": {
            "input_tokens": 11_476,
            "output_tokens": 6_919,
            "requests": 17,
            "source_sha256": TRACE_SHA256,
        },
    }
    output["record_sha256"] = hashlib.sha256(canonical(output)).hexdigest()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-result", type=Path, required=True)
    parser.add_argument("--control-phone", type=Path, required=True)
    parser.add_argument("--control-plan", type=Path, required=True)
    parser.add_argument("--treatment-result", type=Path, required=True)
    parser.add_argument("--treatment-phone", type=Path, required=True)
    parser.add_argument("--treatment-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be an absolute new path")
    try:
        value = analyze(
            args.control_result,
            args.control_phone,
            args.control_plan,
            args.treatment_result,
            args.treatment_phone,
            args.treatment_plan,
        )
        args.output.write_bytes(canonical(value))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"GPU overflow pair analysis failed: {exc}\n")
    print(json.dumps({
        "fleet_energy_change_pct": value["comparison"]["fleet_energy_change_pct"],
        "makespan_change_pct": value["comparison"]["makespan_change_pct"],
        "record_sha256": value["record_sha256"],
        "verdict": value["energy_verdict"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
