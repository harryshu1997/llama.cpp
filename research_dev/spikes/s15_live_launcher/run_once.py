#!/usr/bin/env python3
"""Run one coordinator-triggered physical OP15 B32 launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
S15 = HERE.parent / "s15_runtime_dispatch"
S14 = HERE.parent / "s14_mixed_streaming_scheduler"
sys.path[:0] = [str(HERE), str(S15), str(S14)]

from executor_contract import BOUNDARY_SCHEMA  # noqa: E402
from physical_executor import PhysicalExecutor, PhysicalRouteBinding  # noqa: E402
from power_frontier_policy import BatchDecision, WorkItem  # noqa: E402
from priority_batch_runtime import Launch  # noqa: E402
from route_fixtures import op15_b32_snapshot  # noqa: E402
from route_registry import ReadyRouteRegistry  # noqa: E402
from runtime_dispatch import LaneBinding, MixedDispatchCoordinator  # noqa: E402

import physical_launcher as launcher  # noqa: E402
from prepared_transport import PreparedSubprocessTransport  # noqa: E402


RESULTS = HERE / "results"
REPORT = RESULTS / "result.json"
ROUTE = "op15-gemma-head-0-8"


class RunError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): "sha256:" + sha256(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise RunError("physical execution requires --run")
    if RESULTS.exists():
        raise RunError("results directory already exists")

    cohort, _, request_ids, _ = launcher.load_workload()
    snapshot = op15_b32_snapshot(route_epoch=12)
    if snapshot.profile_id != launcher.EXPECTED_PROFILE:
        raise RunError("runtime route profile does not match the physical launcher")

    transport = PreparedSubprocessTransport(
        (sys.executable, str(HERE / "physical_launcher.py")),
        RESULTS / "transport",
    )
    readiness = transport.ready_metadata
    if readiness.get("cohort_sha256") != launcher.EXPECTED_COHORT \
            or readiness.get("input_manifest_sha256") != launcher.EXPECTED_INPUT:
        transport.terminate()
        raise RunError("prepared launcher workload identity mismatch")
    boot_id = readiness.get("device_boot_id")
    if type(boot_id) is not str or not boot_id:
        transport.terminate()
        raise RunError("prepared launcher has no device boot identity")

    registry = ReadyRouteRegistry()
    registry.install([snapshot])
    binding = PhysicalRouteBinding(
        route_id=ROUTE,
        profile_id=snapshot.profile_id,
        device_id=snapshot.device_id,
        protocol_version=1,
        worker_binary_sha256=launcher.EXPECTED_WORKER,
        worker_generation=1,
        first_session_id=1,
        device_boot_id=boot_id,
        cohort_sha256=launcher.EXPECTED_COHORT,
        input_manifest_sha256=launcher.EXPECTED_INPUT,
        layer_range=(0, 8),
        expected_backend="HTP0",
    )
    executor = PhysicalExecutor((binding,), transport)
    lane = LaneBinding(
        "op15_gemma_head", ROUTE, "phone", timeout_us=5_000_000,
        cohort_sha256=launcher.EXPECTED_COHORT,
        input_manifest_sha256=launcher.EXPECTED_INPUT,
    )
    coordinator = MixedDispatchCoordinator(
        registry, executor, (lane,), queue_capacity=32,
    )

    decisions = []
    final = None
    for request in cohort["requests"]:
        request_id = request["event_id"]
        now_us = request["observed_t_us"]
        item = WorkItem(
            request_id,
            "generation",
            "gemma-4-12b-it-f16",
            "gemma-head-0-8",
            "gemma-4-12b-it-f16|decode|gemma-head-0-8",
            now_us,
            now_us + 5_000_000,
            1,
        )
        admitted = coordinator.admit(item, now_us)
        if admitted.disposition != "queued":
            transport.terminate()
            raise RunError(f"cohort request was not queued: {request_id}")
        decision = coordinator.dispatch(ROUTE, now_us)
        if isinstance(decision, BatchDecision):
            decisions.append({
                "after_request_id": request_id,
                "at_us": now_us,
                "action": decision.action,
                "reason": decision.reason,
                "next_wake_us": decision.next_wake_us,
            })
        elif isinstance(decision, Launch):
            final = decision
            decisions.append({
                "after_request_id": request_id,
                "at_us": now_us,
                "action": "LAUNCH",
                "reason": decision.reason,
                "batch_size": decision.batch_size,
                "request_ids": list(decision.request_ids),
            })
        else:
            transport.terminate()
            raise RunError("runtime returned an unknown decision type")

    if final is None or final.batch_size != 32 or final.reason != "target_batch_ready" \
            or final.request_ids != request_ids:
        raise RunError("coordinator did not produce the exact frozen B32 launch")
    if len(decisions) != 32 or any(value["action"] != "WAIT" for value in decisions[:-1]):
        raise RunError("coordinator did not wait before the complete B32 cohort")
    transport.finalize(30)
    terminal = {request_id: coordinator.terminal_of(request_id) for request_id in request_ids}
    if set(terminal.values()) != {"completed_phone"}:
        raise RunError("physical completion did not own every cohort request")

    paid = json.loads((RESULTS / "raw/launch-1/paid_window.json").read_text(encoding="ascii"))
    report = {
        "schema": "s15-coordinator-physical-b32-v1",
        "verdict": "COORDINATOR_TRIGGERED_PHYSICAL_MECHANICS_PASS_ARRIVAL_FAITHFUL_SLO_BLOCKED",
        "scope": "REAL_OP15_AND_A6000_CORRECTNESS_PLACEMENT_COORDINATOR_MECHANICS_ENERGY_UNKNOWN",
        "energy_scope": "UNKNOWN",
        "cohort_file_sha256": launcher.EXPECTED_COHORT,
        "input_manifest_file_sha256": launcher.EXPECTED_INPUT,
        "inner_cohort_hash": cohort["cohort_hash"],
        "inner_input_manifest_hash": cohort["execution_target"]["input_manifest_hash"],
        "request_ids": list(request_ids),
        "decisions": decisions,
        "terminal_states": terminal,
        "paid_window": paid,
        "prompt_visible_during_preflight": True,
        "arrival_faithful_slo_claim": False,
        "problems": [],
    }
    report["artifact_hashes_before_report"] = artifact_hashes(RESULTS)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps({"verdict": report["verdict"],
                      "paid_elapsed_us": paid["elapsed_us"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
