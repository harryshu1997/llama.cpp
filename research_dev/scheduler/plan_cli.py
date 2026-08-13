#!/usr/bin/env python3
"""Inspect and validate a hash-bound scheduler execution plan."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler._internal.executor import (  # noqa: E402
    ExecutionPlanError,
    load_execution_plan,
)


def field_value(plan, field: str) -> bool | int | str:
    if field == "execution_mode":
        return plan.execution_mode
    if field.startswith("runtime."):
        key = field.removeprefix("runtime.")
        if key not in plan.runtime_bindings:
            raise ExecutionPlanError(f"runtime field is missing: {key}")
        return plan.runtime_bindings[key]
    if field.startswith("artifact.") and field.endswith(".path"):
        role = field[len("artifact."):-len(".path")]
        return plan.artifact(role).path
    if field.startswith("placement."):
        if plan.layer_placement is None:
            raise ExecutionPlanError("plan has no layer placement contract")
        placement = plan.layer_placement
        if field == "placement.cpu_layer_spec":
            return placement.cpu_layer_spec
        if field == "placement.gpu_layer_spec":
            return placement.gpu_layer_spec
        if field == "placement.runtime_gpu_layers":
            return placement.selected.runtime_gpu_layers
        if field == "placement.gpu_total_bytes":
            return placement.selected.gpu_total_bytes
        if field == "placement.cpu_total_bytes":
            return placement.selected.cpu_total_bytes
        if field == "placement.gpu_available_bytes":
            return placement.gpu_capacity.available_bytes
        if field == "placement.cpu_available_bytes":
            return placement.cpu_capacity.available_bytes
        raise ExecutionPlanError(f"unsupported plan field: {field}")
    if plan.offload is None:
        raise ExecutionPlanError("plan has no operator offload contract")
    if field == "offload.policy_id":
        return plan.offload.split.policy_id
    if field == "offload.split_table":
        return plan.offload.split.table
    if field == "offload.max_tokens":
        return plan.offload.split.max_tokens
    if field == "offload.max_columns":
        return plan.offload.split.eligible_columns
    if field == "offload.column_quantum":
        return plan.offload.split.column_quantum
    if field == "offload.alternate_columns":
        return (
            plan.offload.split.alternate_columns[0]
            if plan.offload.split.alternate_columns else 0
        )
    if field == "offload.io_type":
        return plan.offload.split.io_type
    if field == "offload.compute_backend":
        return plan.offload.backend.compute_backend
    if field == "offload.layer_spec":
        return plan.offload.backend.layer_spec
    if field == "offload.session_timeout_s":
        return plan.offload.backend.session_timeout_s
    if field == "offload.max_requests":
        return plan.offload.backend.max_requests
    if field == "offload.bridge_bind":
        return plan.offload.backend.bridge_bind
    if field == "offload.bridge_port":
        return plan.offload.backend.bridge_port
    if field == "offload.allocator":
        return plan.offload.transport.allocator
    if field == "offload.staged_dmabuf":
        return plan.offload.phone_environment()["S41_FFN_STAGED_DMABUF"]
    if field == "offload.resident_weight_bytes":
        value = plan.offload.backend.resident_weight_bytes
        if value is None:
            raise ExecutionPlanError("phone resident weight size is missing")
        return value
    if field == "offload.resident_weight_budget_bytes":
        value = plan.offload.backend.resident_weight_budget_bytes
        if value is None:
            raise ExecutionPlanError("phone resident weight budget is missing")
        return value
    raise ExecutionPlanError(f"unsupported plan field: {field}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--plan", type=Path, required=True)
    validate.add_argument("--mode", required=True)
    get = subparsers.add_parser("get")
    get.add_argument("--plan", type=Path, required=True)
    get.add_argument("--field", required=True)
    args = parser.parse_args()
    try:
        plan = load_execution_plan(args.plan)
        if args.command == "validate":
            if plan.execution_mode != args.mode:
                raise ExecutionPlanError("execution mode differs")
            print(plan.plan_sha256)
        else:
            value = field_value(plan, args.field)
            if type(value) is bool:
                print("1" if value else "0")
            else:
                print(value)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"execution plan check failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
