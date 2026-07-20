#!/usr/bin/env python3
"""Run one arrival-faithful coordinator-triggered physical OP15 B32 launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
LIVE = HERE.parent / "s15_live_launcher"
S15 = HERE.parent / "s15_runtime_dispatch"
S14 = HERE.parent / "s14_mixed_streaming_scheduler"
for path in (LIVE, S15, S14, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import physical_launcher as base  # noqa: E402
from executor_contract import BOUNDARY_SCHEMA  # noqa: E402
from physical_executor import PhysicalExecutor, PhysicalRouteBinding  # noqa: E402
from power_frontier_policy import BatchDecision, WorkItem  # noqa: E402
from prepared_transport import PreparedSubprocessTransport  # noqa: E402
from priority_batch_runtime import Launch  # noqa: E402
from route_fixtures import op15_post_load_b32_snapshot  # noqa: E402
from route_registry import ReadyRouteRegistry  # noqa: E402
from runtime_dispatch import LaneBinding, MixedDispatchCoordinator  # noqa: E402


RESULTS = HERE / "physical_results"
REPORT = RESULTS / "result.json"
ROUTE = base.EXPECTED_ROUTE


class RunError(RuntimeError):
    pass


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): digest(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise RunError("physical execution requires --run")
    if RESULTS.exists():
        raise RunError("physical results already exist")
    cohort, _, request_ids, _ = base.load_workload()
    snapshot = op15_post_load_b32_snapshot()
    b32 = next(point for point in snapshot.config.points if point.batch_size == 32)
    if b32.duration_us != 3_968_367:
        raise RunError("post-load route profile changed")

    transport = PreparedSubprocessTransport(
        (sys.executable, str(HERE / "arrival_launcher.py")),
        RESULTS / "transport",
    )
    readiness = transport.ready_metadata
    boot_id = readiness.get("device_boot_id")
    if readiness.get("cohort_sha256") != base.EXPECTED_COHORT \
            or readiness.get("input_manifest_sha256") != base.EXPECTED_INPUT \
            or type(boot_id) is not str or not boot_id:
        transport.terminate()
        raise RunError("prepared launcher readiness identity failed")

    registry = ReadyRouteRegistry()
    registry.install([snapshot])
    binding = PhysicalRouteBinding(
        route_id=ROUTE,
        profile_id=snapshot.profile_id,
        device_id=snapshot.device_id,
        protocol_version=1,
        worker_binary_sha256=base.EXPECTED_WORKER,
        worker_generation=1,
        first_session_id=1,
        device_boot_id=boot_id,
        cohort_sha256=base.EXPECTED_COHORT,
        input_manifest_sha256=base.EXPECTED_INPUT,
        layer_range=(0, 8),
        expected_backend="HTP0",
    )
    coordinator = MixedDispatchCoordinator(
        registry,
        PhysicalExecutor((binding,), transport),
        (LaneBinding(
            "op15_gemma_head", ROUTE, "phone", timeout_us=4_000_000,
            cohort_sha256=base.EXPECTED_COHORT,
            input_manifest_sha256=base.EXPECTED_INPUT,
        ),),
        queue_capacity=32,
    )

    decisions = []
    final = None
    for value in cohort["requests"]:
        request_id = value["event_id"]
        now_us = value["observed_t_us"]
        item = WorkItem(
            request_id, "generation", "gemma-4-12b-it-f16", "gemma-head-0-8",
            "gemma-4-12b-it-f16|decode|gemma-head-0-8",
            now_us, now_us + 5_000_000, 1,
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
                "predicted_duration_us": decision.predicted_duration_us,
                "request_ids": list(decision.request_ids),
            })
        else:
            transport.terminate()
            raise RunError("runtime returned an unknown decision")
    if final is None or final.batch_size != 32 or final.request_ids != request_ids \
            or final.predicted_duration_us != 3_968_367:
        transport.terminate()
        raise RunError("coordinator did not launch the certified exact B32 cohort")
    transport.finalize(30)
    terminals = {request_id: coordinator.terminal_of(request_id) for request_id in request_ids}
    if set(terminals.values()) != {"completed_phone"}:
        raise RunError("arrival-faithful completion did not own all requests")

    paid = json.loads(
        (RESULTS / "raw/launch-1/paid_window.json").read_text(encoding="ascii")
    )
    transport_record = json.loads(
        (RESULTS / "transport/transport.json").read_text(encoding="ascii")
    )
    launch_us = final.start_us
    earliest_deadline_us = min(
        value["observed_t_us"] + 5_000_000 for value in cohort["requests"]
    )
    logical_finish_us = launch_us + transport_record["elapsed_us"]
    if logical_finish_us > earliest_deadline_us:
        raise RunError("full response missed an admitted request deadline")
    report = {
        "schema": "s15-arrival-faithful-physical-b32-v1",
        "verdict": "ARRIVAL_FAITHFUL_PHYSICAL_B32_PASS",
        "scope": "REAL_OP15_A6000_OBSERVED_ARRIVAL_REPLAY_SYNTHETIC_PAYLOAD_PRIORITY_SLO_ENERGY_UNKNOWN",
        "energy_scope": "UNKNOWN",
        "cohort_sha256": base.EXPECTED_COHORT,
        "input_manifest_sha256": base.EXPECTED_INPUT,
        "profile_id": snapshot.profile_id,
        "profile_duration_us": b32.duration_us,
        "request_ids": list(request_ids),
        "decisions": decisions,
        "terminal_states": terminals,
        "paid_window": paid,
        "transport_elapsed_us": transport_record["elapsed_us"],
        "logical_launch_us": launch_us,
        "logical_finish_us": logical_finish_us,
        "earliest_deadline_us": earliest_deadline_us,
        "deadline_margin_us": earliest_deadline_us - logical_finish_us,
        "prompt_visible_during_preflight": False,
        "arrival_replay_slo_claim": True,
        "problems": [],
    }
    report["artifact_hashes_before_report"] = artifact_hashes(RESULTS)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps({
        "verdict": report["verdict"],
        "transport_elapsed_us": report["transport_elapsed_us"],
        "deadline_margin_us": report["deadline_margin_us"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
