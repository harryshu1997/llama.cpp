#!/usr/bin/env python3
"""Ingest the frozen S14 mix trace plus a synthetic priority/SLO sidecar.

The mix trace carries null deadline/priority fields; BurstGPT and RAGPulse
fields are never read as real deadlines or priorities. The sidecar defined here
is explicitly synthetic (provenance s15-synthetic-sidecar): it assigns a
priority class and a relative deadline by service, and a deterministic head
island. The output is a canonical JSONL decision log that is byte-identical
across processes and PYTHONHASHSEED values.

No latency or energy is claimed. Terminal states are dispatch outcomes.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

_S14 = Path(__file__).resolve().parent.parent / "s14_mixed_streaming_scheduler"
if str(_S14) not in sys.path:
    sys.path.insert(0, str(_S14))

from power_frontier_policy import WorkItem  # noqa: E402

from executor_contract import RecordedExecutor, RecordedOutcome, recorded_key  # noqa: E402
from route_fixtures import op12_head_snapshot, op15_head_snapshot  # noqa: E402
from route_registry import ReadyRouteRegistry  # noqa: E402
from runtime_dispatch import LaneBinding, MixedDispatchCoordinator  # noqa: E402

MECHANICS_INPUT_MANIFEST = "sha256:" + "00" * 32
MECHANICS_COHORT = "sha256:" + "11" * 32


TRACE_PATH = _S14 / "fixtures" / "mix_v1.trace.jsonl"

SIDECAR = {
    "provenance": "s15-synthetic-sidecar",
    "priority_by_service": {
        "api_generation": 1,
        "conversation_generation": 0,
        "rag_qa": 0,
    },
    "deadline_budget_us_by_priority": {0: 300_000, 1: 5_000_000},
    "model_id": "gemma-4-12b-it-f16",
    "service_class": "generation",
    "phone_islands": ["gemma-head-0-8", "gemma-head-0-6"],
}

OP15_ROUTE = "op15-gemma-head-0-8"
OP12_ROUTE = "op12-gemma-head-0-6"
_ISLAND_ROUTE = {"gemma-head-0-8": OP15_ROUTE, "gemma-head-0-6": OP12_ROUTE}


class TraceAdapterError(ValueError):
    pass


@dataclass(frozen=True)
class PlannedRequest:
    event_id: str
    t_us: int
    service: str
    source: str
    priority_class: int
    deadline_us: int
    island_id: str
    target_route: str


def load_trace(path: Path = TRACE_PATH) -> list[dict]:
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise TraceAdapterError(f"duplicate key {key!r} in trace row")
            result[key] = value
        return result

    rows: list[dict] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if not line:
            continue
        rows.append(json.loads(line, object_pairs_hook=no_duplicates))
    ordered = sorted(rows, key=lambda row: (row["t_us"], row["event_id"]))
    seen = set()
    for row in ordered:
        if row["event_id"] in seen:
            raise TraceAdapterError(f"duplicate event_id {row['event_id']!r}")
        seen.add(row["event_id"])
        if row.get("deadline_us") is not None or row.get("priority_class") is not None:
            raise TraceAdapterError("trace already carries a deadline or priority")
    return ordered


def apply_sidecar(rows: list[dict]) -> list[PlannedRequest]:
    planned: list[PlannedRequest] = []
    islands = SIDECAR["phone_islands"]
    for index, row in enumerate(rows):
        service = row["service"]
        if service not in SIDECAR["priority_by_service"]:
            raise TraceAdapterError(f"no synthetic priority for service {service!r}")
        priority = SIDECAR["priority_by_service"][service]
        budget = SIDECAR["deadline_budget_us_by_priority"][priority]
        island = islands[index % len(islands)]
        planned.append(PlannedRequest(
            event_id=row["event_id"],
            t_us=row["t_us"],
            service=service,
            source=row["source"],
            priority_class=priority,
            deadline_us=row["t_us"] + budget,
            island_id=island,
            target_route=_ISLAND_ROUTE[island],
        ))
    return planned


def to_work_item(planned: PlannedRequest) -> WorkItem:
    return WorkItem(
        request_id=planned.event_id,
        service_class=SIDECAR["service_class"],
        model_id=SIDECAR["model_id"],
        island_id=planned.island_id,
        compatibility_key=f"{SIDECAR['model_id']}|decode|{planned.island_id}",
        arrival_us=planned.t_us,
        deadline_us=planned.deadline_us,
        priority_class=planned.priority_class,
    )


def build_registry() -> tuple[ReadyRouteRegistry, tuple[LaneBinding, ...]]:
    op15 = op15_head_snapshot()
    op12 = op12_head_snapshot()
    registry = ReadyRouteRegistry()
    registry.install([op15, op12])
    lanes = (
        LaneBinding("op15_gemma_head", OP15_ROUTE, "phone", timeout_us=4_000_000, cohort_sha256=MECHANICS_COHORT, input_manifest_sha256=MECHANICS_INPUT_MANIFEST),
        LaneBinding("op12_gemma_head", OP12_ROUTE, "phone", timeout_us=4_000_000, cohort_sha256=MECHANICS_COHORT, input_manifest_sha256=MECHANICS_INPUT_MANIFEST),
    )
    return registry, lanes


def build_executor(registry: ReadyRouteRegistry, planned: list[PlannedRequest]) -> RecordedExecutor:
    """Record a completed outcome for every phone-feasible request only."""
    script: dict = {}
    for request in planned:
        snapshot = registry.get(request.target_route)
        duration = min(point.duration_us for point in snapshot.config.points)
        if request.t_us + duration > request.deadline_us:
            continue  # infeasible -> served by fallback, no phone launch recorded
        script[recorded_key(request.target_route, (request.event_id,))] = RecordedOutcome(
            outcome="completed",
            finish_delay_us=duration,
            profile_id=snapshot.profile_id,
            route_epoch=snapshot.route_epoch,
            residency_epoch=snapshot.residency_epoch,
            device_boot_epoch=snapshot.device_boot_epoch,
            cohort_sha256=MECHANICS_COHORT, input_manifest_sha256=MECHANICS_INPUT_MANIFEST,
        )
    return RecordedExecutor(script)


def replay() -> list[dict]:
    rows = load_trace()
    planned = apply_sidecar(rows)
    registry, lanes = build_registry()
    executor = build_executor(registry, planned)
    coordinator = MixedDispatchCoordinator(registry, executor, lanes, queue_capacity=64)
    decisions: list[dict] = []
    for request in planned:
        item = to_work_item(request)
        admit = coordinator.admit(item, request.t_us)
        if admit.disposition == "queued":
            coordinator.dispatch(admit.route_id, request.t_us)
        terminal = coordinator.terminal_of(request.event_id)
        decisions.append({
            "event_id": request.event_id,
            "t_us": request.t_us,
            "service": request.service,
            "source": request.source,
            "priority_class": request.priority_class,
            "priority_provenance": SIDECAR["provenance"],
            "deadline_us": request.deadline_us,
            "deadline_provenance": SIDECAR["provenance"],
            "model_id": SIDECAR["model_id"],
            "island_id": request.island_id,
            "target_route": request.target_route,
            "disposition": admit.disposition,
            "reason": admit.reason,
            "terminal_state": terminal,
        })
    coordinator.assert_conservation([request.event_id for request in planned])
    return decisions


def emit_log(decisions: list[dict]) -> str:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in decisions
    )


def summarize(decisions: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for row in decisions:
        counts[row["terminal_state"]] = counts.get(row["terminal_state"], 0) + 1
    return {
        "schema": "s15-decision-log-summary-v1",
        "n_requests": len(decisions),
        "terminal_counts": dict(sorted(counts.items())),
        "verdict": "RUNTIME_DISPATCH_MECHANICS_PASS_PHYSICAL_EXECUTION_NOT_RUN",
    }


def main() -> int:
    decisions = replay()
    sys.stdout.write(emit_log(decisions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
