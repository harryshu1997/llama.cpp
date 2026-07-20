#!/usr/bin/env python3
"""Deterministic bounded virtual-queue replay for the S12 mechanics gate."""

from __future__ import annotations

import argparse
import heapq
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from policies import (
    POLICIES,
    background_hbm_at,
    causal_batch_size,
    largest_supported_batch,
    memory_admission_choice,
    offline_server_plan,
)
from profile_coverage import prepare_trace
from s12lib import (
    MAX_SAFE_INT,
    MIB,
    S12Error,
    canonical_json,
    load_json,
    nearest_rank,
    profile_rows_by_batch,
    require_bool,
    require_exact_keys,
    require_int,
    require_str,
    read_jsonl_snapshot,
    sha256_object,
    validate_profile,
    write_canonical,
)


PR_COMPLETE = 0
PR_HBM = 1
PR_ARRIVAL = 2
PR_FLUSH = 3

TERMINALS = (
    "completed_server",
    "completed_phone",
    "rejected_queue_full",
    "timed_out",
    "unprofiled",
)


@dataclass
class RequestState:
    event_id: str
    arrival_us: int
    claim_scope: str
    profile_eligible: bool
    terminal: str | None = None
    start_us: int | None = None
    finish_us: int | None = None
    route: str | None = None


def _resolve(config_path: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (config_path.parent / p).resolve()


def validate_config(config: Any) -> dict[str, Any]:
    cfg = require_exact_keys(
        "config",
        config,
        {
            "background_hbm_mib_timeline",
            "batch_hold_us",
            "horizon_us",
            "offline_max_requests",
            "phone_queue_limit",
            "phone_residency",
            "policies",
            "profile_path",
            "schema",
            "server_hbm_capacity_mib",
            "server_queue_limit",
            "trace_mode",
            "trace_path",
        },
    )
    require_str("config.schema", cfg["schema"], {"s12-vq-config-v1"})
    require_str(
        "config.trace_mode",
        cfg["trace_mode"],
        {"strict_real", "semi_synthetic_shape_shadow"},
    )
    for key in (
        "batch_hold_us",
        "horizon_us",
        "phone_queue_limit",
        "server_hbm_capacity_mib",
        "server_queue_limit",
    ):
        require_int(f"config.{key}", cfg[key], 1, MAX_SAFE_INT)
    require_int("config.offline_max_requests", cfg["offline_max_requests"], 1, 24)
    require_str("config.profile_path", cfg["profile_path"])
    require_str("config.trace_path", cfg["trace_path"])
    policies = cfg["policies"]
    if type(policies) is not list or not policies:
        raise S12Error("config.policies: expected non-empty array")
    if any(type(policy) is not str or policy not in POLICIES for policy in policies):
        raise S12Error("config.policies: unsupported policy")
    if len(policies) != len(set(policies)):
        raise S12Error("config.policies: duplicate policy")
    residency = require_exact_keys(
        "config.phone_residency",
        cfg["phone_residency"],
        {"ready", "weight_bytes"},
    )
    require_bool("config.phone_residency.ready", residency["ready"])
    require_int("config.phone_residency.weight_bytes", residency["weight_bytes"], 1)
    timeline = cfg["background_hbm_mib_timeline"]
    if type(timeline) is not list or not timeline:
        raise S12Error("config.background_hbm_mib_timeline: expected non-empty array")
    previous = -1
    for index, point in enumerate(timeline):
        point = require_exact_keys(
            f"config.background_hbm_mib_timeline[{index}]",
            point,
            {"t_us", "used_mib"},
        )
        t_us = require_int(f"config.background_hbm_mib_timeline[{index}].t_us", point["t_us"])
        used_mib = require_int(
            f"config.background_hbm_mib_timeline[{index}].used_mib",
            point["used_mib"],
            0,
            cfg["server_hbm_capacity_mib"],
        )
        if t_us <= previous:
            raise S12Error("config.background_hbm_mib_timeline: timestamps must increase")
        if index == 0 and t_us != 0:
            raise S12Error("config.background_hbm_mib_timeline: first timestamp must be zero")
        previous = t_us
        if used_mib > cfg["server_hbm_capacity_mib"]:
            raise S12Error("config.background_hbm_mib_timeline: background exceeds capacity")
    return cfg


class CausalReplay:
    def __init__(
        self,
        policy: str,
        cfg: dict[str, Any],
        coverage: dict[str, Any],
        rows: dict[int, dict[str, Any]],
    ):
        self.policy = policy
        self.cfg = cfg
        self.coverage = coverage
        self.rows = rows
        self.supported = tuple(sorted(rows))
        self.now = 0
        self.sequence = 0
        self.heap: list[tuple[int, int, int, str, Any]] = []
        self.requests = [
            RequestState(
                event_id=req["event_id"],
                arrival_us=req["arrival_us"],
                claim_scope=req["claim_scope"],
                profile_eligible=req["profile_eligible"],
            )
            for req in coverage["prepared_requests"]
        ]
        self.by_id = {req.event_id: req for req in self.requests}
        self.queue: list[RequestState] = []
        self.server_busy = False
        self.phone_busy = False
        self.active_hbm_mib = 0
        self.decisions: list[dict[str, Any]] = []
        self.queue_depth_max = 0
        self.peak_a6000_hbm_mib = 0
        self.activation_bytes = 0
        self.phone_route_batches = 0
        self.server_route_batches = 0
        self.memory_server_infeasible_events = 0
        self.memory_phone_selections = 0
        for point in cfg["background_hbm_mib_timeline"]:
            self.push(point["t_us"], PR_HBM, "hbm_change", point)
        for req in self.requests:
            self.push(req.arrival_us, PR_ARRIVAL, "arrival", req.event_id)

    def push(self, t_us: int, priority: int, kind: str, payload: Any) -> None:
        require_int("event timestamp", t_us)
        heapq.heappush(self.heap, (t_us, priority, self.sequence, kind, payload))
        self.sequence += 1

    def available_hbm_mib(self) -> int:
        return self.cfg["server_hbm_capacity_mib"] - background_hbm_at(
            self.now,
            self.cfg["background_hbm_mib_timeline"],
        )

    def _finish(self, req: RequestState, terminal: str, finish_us: int | None = None) -> None:
        if terminal not in TERMINALS:
            raise AssertionError(f"unknown terminal {terminal}")
        if req.terminal is not None:
            raise AssertionError(f"request {req.event_id} terminalized twice")
        req.terminal = terminal
        req.finish_us = self.now if finish_us is None else finish_us

    def on_arrival(self, request_id: str) -> None:
        req = self.by_id[request_id]
        if not req.profile_eligible:
            self._finish(req, "unprofiled")
            return
        queue_limit = (
            self.cfg["phone_queue_limit"]
            if self.policy == "fixed_phone"
            else self.cfg["server_queue_limit"]
        )
        if len(self.queue) >= queue_limit:
            self._finish(req, "rejected_queue_full")
            return
        self.queue.append(req)
        self.queue_depth_max = max(self.queue_depth_max, len(self.queue))
        self.push(
            req.arrival_us + self.cfg["batch_hold_us"],
            PR_FLUSH,
            "flush",
            req.event_id,
        )

    def on_complete(self, payload: dict[str, Any]) -> None:
        route = payload["route"]
        if not self.server_busy:
            raise AssertionError("completion without server reservation")
        self.server_busy = False
        if route == "A0_OP15":
            if not self.phone_busy:
                raise AssertionError("phone completion without phone reservation")
            self.phone_busy = False
        self.active_hbm_mib = 0
        terminal = "completed_phone" if route == "A0_OP15" else "completed_server"
        for request_id in payload["request_ids"]:
            self._finish(self.by_id[request_id], terminal)

    def _oldest_ready(self) -> bool:
        if not self.queue:
            return False
        return (
            self.now >= self.queue[0].arrival_us + self.cfg["batch_hold_us"]
            or len(self.queue) >= self.supported[-1]
        )

    def _route_choice(self) -> tuple[str, int, bool, str] | None:
        queue_length = len(self.queue)
        oldest_ready = self._oldest_ready()
        available = self.available_hbm_mib()
        if self.policy == "causal_server_batch":
            requested = causal_batch_size(queue_length, oldest_ready, self.supported)
            if requested is None:
                return None
            for batch in reversed([b for b in self.supported if b <= requested]):
                if self.rows[batch]["control_ready_hbm_mib"] <= available:
                    return "SERVER_ONLY", batch, True, "CAUSAL_SERVER_BATCH"
            return None
        if self.policy == "fixed_phone":
            requested = causal_batch_size(queue_length, oldest_ready, self.supported)
            if requested is None or not self.cfg["phone_residency"]["ready"]:
                return None
            for batch in reversed([b for b in self.supported if b <= requested]):
                if self.rows[batch]["phone_route_ready_hbm_mib"] <= available:
                    server_feasible = self.rows[batch]["control_ready_hbm_mib"] <= available
                    return "A0_OP15", batch, server_feasible, "FIXED_PHONE"
            return None
        if self.policy == "memory_admission_triggered":
            choice = memory_admission_choice(
                queue_length,
                oldest_ready,
                self.supported,
                available,
                self.rows,
                self.cfg["phone_residency"]["ready"],
            )
            if choice.reason in ("SERVER_ADMISSION_INFEASIBLE", "NO_ROUTE_FITS_HBM"):
                self.memory_server_infeasible_events += 1
            if choice.route is None or choice.batch_size is None:
                return None
            if choice.route == "A0_OP15":
                if choice.server_admission_feasible:
                    raise AssertionError("memory policy selected phone while server feasible")
                self.memory_phone_selections += 1
            return (
                choice.route,
                choice.batch_size,
                choice.server_admission_feasible,
                choice.reason,
            )
        raise AssertionError(f"causal replay received unsupported policy {self.policy}")

    def try_dispatch(self) -> None:
        if self.server_busy or not self.queue:
            return
        choice = self._route_choice()
        if choice is None:
            return
        route, batch, server_feasible, reason = choice
        if route == "A0_OP15" and self.phone_busy:
            return
        row = self.rows[batch]
        route_hbm = (
            row["phone_route_ready_hbm_mib"]
            if route == "A0_OP15"
            else row["control_ready_hbm_mib"]
        )
        if route_hbm > self.available_hbm_mib():
            raise AssertionError("dispatch exceeds current A6000 admission")
        selected = self.queue[:batch]
        del self.queue[:batch]
        duration = row["phone_route_group_us"] if route == "A0_OP15" else row["server_group_us"]
        finish = self.now + duration
        if finish > self.cfg["horizon_us"]:
            self.queue = selected + self.queue
            return
        self.server_busy = True
        self.phone_busy = route == "A0_OP15"
        self.active_hbm_mib = route_hbm
        background = background_hbm_at(
            self.now,
            self.cfg["background_hbm_mib_timeline"],
        )
        self.peak_a6000_hbm_mib = max(
            self.peak_a6000_hbm_mib,
            background + route_hbm,
        )
        request_ids = []
        for req in selected:
            req.start_us = self.now
            req.route = route
            request_ids.append(req.event_id)
        activation_bytes = row["activation_bytes"] if route == "A0_OP15" else 0
        if route == "A0_OP15":
            self.phone_route_batches += 1
            self.activation_bytes += activation_bytes
        else:
            self.server_route_batches += 1
        decision = {
            "activation_bytes": activation_bytes,
            "available_hbm_mib": self.available_hbm_mib(),
            "batch_size": batch,
            "control_hbm_mib": row["control_ready_hbm_mib"],
            "finish_us": finish,
            "reason": reason,
            "request_ids": request_ids,
            "route": route,
            "route_hbm_mib": route_hbm,
            "server_admission_feasible": server_feasible,
            "server_hbm_relief_mib": row["server_hbm_relief_mib"]
            if route == "A0_OP15"
            else 0,
            "start_us": self.now,
        }
        if (
            self.policy == "memory_admission_triggered"
            and route == "A0_OP15"
            and decision["server_admission_feasible"]
        ):
            raise AssertionError("memory phone dispatch violated server-first invariant")
        self.decisions.append(decision)
        self.push(
            finish,
            PR_COMPLETE,
            "complete",
            {"request_ids": request_ids, "route": route},
        )

    def _check_invariants(self) -> None:
        if self.active_hbm_mib:
            total = (
                background_hbm_at(self.now, self.cfg["background_hbm_mib_timeline"])
                + self.active_hbm_mib
            )
            if total > self.cfg["server_hbm_capacity_mib"]:
                raise S12Error("background HBM change oversubscribed an active route")
        if self.phone_busy and not self.server_busy:
            raise AssertionError("phone route must reserve the shared A6000 lane")
        if len(self.queue) > max(
            self.cfg["server_queue_limit"],
            self.cfg["phone_queue_limit"],
        ):
            raise AssertionError("queue bound exceeded")

    def run(self) -> dict[str, Any]:
        while self.heap:
            t_us = self.heap[0][0]
            if t_us > self.cfg["horizon_us"]:
                break
            self.now = t_us
            while self.heap and self.heap[0][0] == t_us:
                _, _, _, kind, payload = heapq.heappop(self.heap)
                if kind == "arrival":
                    self.on_arrival(payload)
                elif kind == "complete":
                    self.on_complete(payload)
                elif kind == "hbm_change":
                    self.peak_a6000_hbm_mib = max(
                        self.peak_a6000_hbm_mib,
                        payload["used_mib"] + self.active_hbm_mib,
                    )
                elif kind == "flush":
                    pass
                else:
                    raise AssertionError(f"unknown event {kind}")
                self._check_invariants()
            self.try_dispatch()
            self._check_invariants()

        self.now = self.cfg["horizon_us"]
        for req in self.requests:
            if req.terminal is None:
                self._finish(req, "timed_out")
        self.server_busy = False
        self.phone_busy = False
        self.active_hbm_mib = 0
        return build_result(
            policy=self.policy,
            cfg=self.cfg,
            coverage=self.coverage,
            requests=self.requests,
            decisions=self.decisions,
            queue_depth_max=self.queue_depth_max,
            peak_a6000_hbm_mib=self.peak_a6000_hbm_mib,
            activation_bytes=self.activation_bytes,
            phone_route_batches=self.phone_route_batches,
            server_route_batches=self.server_route_batches,
            memory_server_infeasible_events=self.memory_server_infeasible_events,
            memory_phone_selections=self.memory_phone_selections,
            clairvoyant=False,
        )


def run_offline(
    cfg: dict[str, Any],
    coverage: dict[str, Any],
    rows: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    requests = [
        RequestState(
            event_id=req["event_id"],
            arrival_us=req["arrival_us"],
            claim_scope=req["claim_scope"],
            profile_eligible=req["profile_eligible"],
        )
        for req in coverage["prepared_requests"]
    ]
    eligible = [req for req in requests if req.profile_eligible]
    for req in requests:
        if not req.profile_eligible:
            req.terminal = "unprofiled"
            req.finish_us = req.arrival_us
    groups = offline_server_plan(
        [req.arrival_us for req in eligible],
        rows,
        cfg["server_hbm_capacity_mib"],
        cfg["background_hbm_mib_timeline"],
        cfg["server_queue_limit"],
        cfg["horizon_us"],
        cfg["offline_max_requests"],
    )
    decisions = []
    cursor = 0
    peak_hbm = max(point["used_mib"] for point in cfg["background_hbm_mib_timeline"])
    for group in groups:
        batch = group["batch_size"]
        selected = eligible[cursor : cursor + batch]
        cursor += batch
        row = rows[batch]
        request_ids = []
        for req in selected:
            req.start_us = group["start_us"]
            req.finish_us = group["finish_us"]
            req.route = "SERVER_ONLY"
            req.terminal = "completed_server"
            request_ids.append(req.event_id)
        active_times = [group["start_us"]]
        active_times.extend(
            point["t_us"]
            for point in cfg["background_hbm_mib_timeline"]
            if group["start_us"] < point["t_us"] < group["finish_us"]
        )
        peak_hbm = max(
            peak_hbm,
            max(
                background_hbm_at(t_us, cfg["background_hbm_mib_timeline"])
                + row["control_ready_hbm_mib"]
                for t_us in active_times
            ),
        )
        decisions.append(
            {
                "activation_bytes": 0,
                "available_hbm_mib": cfg["server_hbm_capacity_mib"]
                - background_hbm_at(group["start_us"], cfg["background_hbm_mib_timeline"]),
                "batch_size": batch,
                "control_hbm_mib": row["control_ready_hbm_mib"],
                "finish_us": group["finish_us"],
                "reason": "CLAIRVOYANT_OFFLINE_PARTITION",
                "request_ids": request_ids,
                "route": "SERVER_ONLY",
                "route_hbm_mib": row["control_ready_hbm_mib"],
                "server_admission_feasible": True,
                "server_hbm_relief_mib": 0,
                "start_us": group["start_us"],
            }
        )
    starts = {request_id: decision["start_us"] for decision in decisions for request_id in decision["request_ids"]}
    queue_depth_max = 0
    for now in [req.arrival_us for req in eligible]:
        queue_depth_max = max(
            queue_depth_max,
            sum(1 for req in eligible if req.arrival_us <= now and starts[req.event_id] > now),
        )
    return build_result(
        policy="server_only_optimized",
        cfg=cfg,
        coverage=coverage,
        requests=requests,
        decisions=decisions,
        queue_depth_max=queue_depth_max,
        peak_a6000_hbm_mib=peak_hbm,
        activation_bytes=0,
        phone_route_batches=0,
        server_route_batches=len(groups),
        memory_server_infeasible_events=0,
        memory_phone_selections=0,
        clairvoyant=True,
    )


def build_result(
    *,
    policy: str,
    cfg: dict[str, Any],
    coverage: dict[str, Any],
    requests: list[RequestState],
    decisions: list[dict[str, Any]],
    queue_depth_max: int,
    peak_a6000_hbm_mib: int,
    activation_bytes: int,
    phone_route_batches: int,
    server_route_batches: int,
    memory_server_infeasible_events: int,
    memory_phone_selections: int,
    clairvoyant: bool,
) -> dict[str, Any]:
    counts = {terminal: 0 for terminal in TERMINALS}
    queue_delays = []
    completion_latencies = []
    outcomes = []
    for req in requests:
        if req.terminal is None:
            raise AssertionError(f"request {req.event_id} has no terminal")
        counts[req.terminal] += 1
        if req.start_us is not None:
            queue_delays.append(req.start_us - req.arrival_us)
        if req.finish_us is not None and req.terminal.startswith("completed_"):
            completion_latencies.append(req.finish_us - req.arrival_us)
        outcomes.append(
            {
                "arrival_us": req.arrival_us,
                "event_id": req.event_id,
                "finish_us": req.finish_us,
                "route": req.route,
                "start_us": req.start_us,
                "terminal": req.terminal,
            }
        )
    if sum(counts.values()) != len(requests):
        raise AssertionError("terminal conservation failed")
    if policy == "memory_admission_triggered":
        for decision in decisions:
            if decision["route"] == "A0_OP15" and decision["server_admission_feasible"]:
                raise AssertionError("memory policy phone decision was not admission-triggered")
    completed = counts["completed_server"] + counts["completed_phone"]
    completed_finishes = [
        req.finish_us
        for req in requests
        if req.terminal is not None
        and req.terminal.startswith("completed_")
        and req.finish_us is not None
    ]
    completed_arrivals = [
        req.arrival_us
        for req in requests
        if req.terminal is not None and req.terminal.startswith("completed_")
    ]
    makespan_us = (
        max(completed_finishes) - min(completed_arrivals)
        if completed_finishes and completed_arrivals
        else None
    )
    result = {
        "activation_bytes_total": activation_bytes,
        "activation_formula": "B*(28+3)*3840*4",
        "claim_scope": coverage["claim_scope"],
        "clairvoyant": clairvoyant,
        "is_offline_upper_bound": clairvoyant,
        "completed_generated_tokens": completed * 4,
        "completed_phone": counts["completed_phone"],
        "completed_prompt_tokens": completed * 28,
        "completed_requests": completed,
        "completed_server": counts["completed_server"],
        "completed_throughput": (
            {
                "denominator_us": makespan_us,
                "numerator_requests_x1000000": completed * 1000000,
            }
            if makespan_us
            else None
        ),
        "completion_latency_p50_us": nearest_rank(completion_latencies, 1, 2),
        "completion_latency_p95_us": nearest_rank(completion_latencies, 19, 20),
        "decisions": decisions,
        "energy": {
            "phone_nj": None,
            "server_gpu_board_nj": None,
            "status": "NOT_RUN",
            "total_wall_nj": None,
        },
        "memory_phone_selections": memory_phone_selections,
        "memory_server_infeasible_events": memory_server_infeasible_events,
        "makespan_us": makespan_us,
        "outcomes": outcomes,
        "peak_a6000_hbm_bytes": peak_a6000_hbm_mib * MIB,
        "peak_a6000_hbm_mib": peak_a6000_hbm_mib,
        "phone_residency_ready": cfg["phone_residency"]["ready"],
        "phone_resident_weight_bytes": cfg["phone_residency"]["weight_bytes"],
        "phone_route_batches": phone_route_batches,
        "phone_route_relief_mib_observations": [
            decision["server_hbm_relief_mib"]
            for decision in decisions
            if decision["route"] == "A0_OP15"
        ],
        "policy": policy,
        "profiled_requests": coverage["profiled_requests"],
        "queue_delay_p50_us": nearest_rank(queue_delays, 1, 2),
        "queue_delay_p95_us": nearest_rank(queue_delays, 19, 20),
        "queue_depth_max": queue_depth_max,
        "rejected_queue_full": counts["rejected_queue_full"],
        "rejected_requests": counts["rejected_queue_full"],
        "requests": len(requests),
        "schema": "s12-vq-policy-result-v1",
        "server_hbm_capacity_bytes": cfg["server_hbm_capacity_mib"] * MIB,
        "server_hbm_capacity_mib": cfg["server_hbm_capacity_mib"],
        "server_route_batches": server_route_batches,
        "terminal_conservation": sum(counts.values()),
        "timed_out": counts["timed_out"],
        "unprofiled": counts["unprofiled"],
        "upper_bound_scope": (
            "CLAIRVOYANT_WITHIN_FROZEN_FCFS_SINGLE_LANE_DOMAIN"
            if clairvoyant
            else None
        ),
    }
    result["result_digest"] = sha256_object(result)
    return result


def run_config(config_path: str | Path, selected_policies: list[str] | None = None) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    cfg = validate_config(load_json(config_file))
    profile_file = _resolve(config_file, cfg["profile_path"])
    trace_file = _resolve(config_file, cfg["trace_path"])
    profile = validate_profile(load_json(profile_file))
    if cfg["phone_residency"]["weight_bytes"] != profile["model"]["phone_weight_bytes"]:
        raise S12Error("config.phone_residency.weight_bytes does not match profile")
    trace_records, trace_digest = read_jsonl_snapshot(trace_file)
    coverage = prepare_trace(trace_records, profile, cfg["trace_mode"])
    rows = profile_rows_by_batch(profile)
    policies = cfg["policies"] if selected_policies is None else selected_policies
    if not policies or len(policies) != len(set(policies)):
        raise S12Error("selected policies must be non-empty and unique")
    if any(policy not in cfg["policies"] for policy in policies):
        raise S12Error("selected policy is not frozen in config")
    results = []
    for policy in policies:
        if policy == "server_only_optimized":
            results.append(run_offline(cfg, coverage, rows))
        else:
            results.append(CausalReplay(policy, cfg, coverage, rows).run())
    manifest = {
        "claim_scope": coverage["claim_scope"],
        "config_digest": sha256_object(cfg),
        "config_path": str(config_file),
        "coverage_digest": coverage["coverage_digest"],
        "energy_status": "NOT_RUN",
        "policy_results": results,
        "profile_digest": sha256_object(profile),
        "profile_path": str(profile_file),
        "schema": "s12-vq-replay-v1",
        "trace_mode": cfg["trace_mode"],
        "trace_path": str(trace_file),
        "trace_sha256": trace_digest,
    }
    manifest["deterministic_replay_sha256"] = sha256_object(results)
    return manifest


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out")
    parser.add_argument("--policy", action="append", choices=POLICIES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = run_config(args.config, args.policy)
        if args.out:
            write_canonical(args.out, result)
        else:
            print(canonical_json(result))
        return 0
    except S12Error as exc:
        print(f"VQ_SIM_FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
