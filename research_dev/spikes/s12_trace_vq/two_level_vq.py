#!/usr/bin/env python3
"""Deterministic two-level mixed-residency scheduler mechanics replay."""

from __future__ import annotations

import argparse
import heapq
import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from s12lib import (
    MAX_SAFE_INT,
    S12Error,
    canonical_json,
    parse_jsonl_bytes,
    read_bytes_once,
    require_exact_keys,
    require_int,
    require_str,
    sha256_bytes,
    sha256_object,
    strict_json_loads,
    write_canonical,
)


SCOPE = "SYNTHETIC_MIXED_RESIDENCY_MECHANICS_ONLY"
ENERGY_STATUS = "NOT_RUN"
POLICIES = (
    "server_only",
    "fixed_static_phone",
    "static_two_phone",
    "dynamic_two_level",
)
REPLICA_STATES = {
    "RECEIVING",
    "VERIFYING",
    "PREPARING",
    "READY",
    "LEASED",
    "DRAINING",
    "EVICTING",
}
LEGAL_REPLICA_TRANSITIONS = {
    "RECEIVING": {"VERIFYING", "EVICTING"},
    "VERIFYING": {"PREPARING", "EVICTING"},
    "PREPARING": {"READY", "EVICTING"},
    "READY": {"LEASED", "DRAINING", "EVICTING"},
    "LEASED": {"READY", "DRAINING"},
    "DRAINING": {"EVICTING"},
    "EVICTING": set(),
}
TERMINALS = (
    "completed_server",
    "completed_phone",
    "tardy_server",
    "tardy_phone",
    "rejected_queue_full",
    "timed_out",
)


def _decode_ascii_json(data: bytes, source: str) -> Any:
    try:
        return strict_json_loads(data.decode("ascii"))
    except UnicodeError as exc:
        raise S12Error(f"{source}: expected ASCII JSON: {exc}") from exc


def _require_array(name: str, value: Any, nonempty: bool = True) -> list[Any]:
    if type(value) is not list or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise S12Error(f"{name}: expected {qualifier}array")
    return value


def _safe_add(name: str, left: int, right: int) -> int:
    require_int(f"{name}.left", left)
    require_int(f"{name}.right", right)
    result = left + right
    if result > MAX_SAFE_INT:
        raise S12Error(f"{name}: integer overflow")
    return result


def _require_unique_strings(
    name: str,
    values: Any,
    allowed: set[str] | None = None,
    nonempty: bool = True,
) -> list[str]:
    array = _require_array(name, values, nonempty=nonempty)
    result = [require_str(f"{name}[{index}]", value, allowed) for index, value in enumerate(array)]
    if len(set(result)) != len(result):
        raise S12Error(f"{name}: duplicate value")
    return result


def validate_profile(raw: Any) -> dict[str, Any]:
    profile = require_exact_keys(
        "profile",
        raw,
        {"schema", "scope", "energy_status", "islands"},
    )
    require_str("profile.schema", profile["schema"], {"s12-two-model-mechanics-profile-v1"})
    require_str("profile.scope", profile["scope"], {SCOPE})
    require_str("profile.energy_status", profile["energy_status"], {ENERGY_STATUS})
    islands = _require_array("profile.islands", profile["islands"])
    seen_islands: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for index, raw_island in enumerate(islands):
        name = f"profile.islands[{index}]"
        island = require_exact_keys(
            name,
            raw_island,
            {
                "model_id",
                "island_id",
                "weight_bytes",
                "weight_identity",
                "synthetic_server_full_us",
                "synthetic_server_tail_us",
                "device_routes",
            },
        )
        model_id = require_str(f"{name}.model_id", island["model_id"])
        island_id = require_str(f"{name}.island_id", island["island_id"])
        if island_id in seen_islands:
            raise S12Error(f"{name}.island_id: duplicate {island_id!r}")
        seen_islands.add(island_id)
        require_int(f"{name}.weight_bytes", island["weight_bytes"], 1)
        require_str(f"{name}.weight_identity", island["weight_identity"])
        require_int(f"{name}.synthetic_server_full_us", island["synthetic_server_full_us"], 1)
        require_int(f"{name}.synthetic_server_tail_us", island["synthetic_server_tail_us"], 1)
        routes = _require_array(f"{name}.device_routes", island["device_routes"])
        route_devices: set[str] = set()
        for route_index, raw_route in enumerate(routes):
            route_name = f"{name}.device_routes[{route_index}]"
            route = require_exact_keys(
                route_name,
                raw_route,
                {
                    "device_id",
                    "backend",
                    "correctness_status",
                    "synthetic_phone_us",
                    "transfer_us",
                    "verify_us",
                    "prepare_us",
                },
            )
            device_id = require_str(f"{route_name}.device_id", route["device_id"])
            if device_id in route_devices:
                raise S12Error(f"{route_name}.device_id: duplicate {device_id!r}")
            route_devices.add(device_id)
            require_str(f"{route_name}.backend", route["backend"])
            require_str(
                f"{route_name}.correctness_status",
                route["correctness_status"],
                {"ASSUMED_SYNTHETIC_ONLY"},
            )
            for field_name in ("synthetic_phone_us", "transfer_us", "verify_us", "prepare_us"):
                require_int(f"{route_name}.{field_name}", route[field_name], 1)
        pair = (model_id, island_id)
        if pair in seen_pairs:
            raise S12Error(f"{name}: duplicate model/island pair")
        seen_pairs.add(pair)
    return profile


def validate_config(raw: Any) -> dict[str, Any]:
    config = require_exact_keys(
        "config",
        raw,
        {
            "schema",
            "scope",
            "energy_status",
            "horizon_us",
            "queue_limit",
            "server_slots",
            "policies",
            "profile_path",
            "trace_path",
            "devices",
            "scheduler",
        },
    )
    require_str("config.schema", config["schema"], {"s12-two-level-config-v1"})
    require_str("config.scope", config["scope"], {SCOPE})
    require_str("config.energy_status", config["energy_status"], {ENERGY_STATUS})
    require_int("config.horizon_us", config["horizon_us"], 1)
    require_int("config.queue_limit", config["queue_limit"], 1)
    if require_int("config.server_slots", config["server_slots"], 1) != 1:
        raise S12Error("config.server_slots: V2a requires exactly one")
    policies = _require_unique_strings("config.policies", config["policies"], set(POLICIES))
    if set(policies) != set(POLICIES):
        raise S12Error(f"config.policies: expected exactly {list(POLICIES)!r}")
    require_str("config.profile_path", config["profile_path"])
    require_str("config.trace_path", config["trace_path"])

    devices = _require_array("config.devices", config["devices"])
    seen_devices: set[str] = set()
    for index, raw_device in enumerate(devices):
        name = f"config.devices[{index}]"
        device = require_exact_keys(
            name,
            raw_device,
            {
                "device_id",
                "boot_epoch",
                "capacity_bytes",
                "activation_slots",
                "prefetch_queue_limit",
                "static_islands",
                "dynamic_initial_islands",
            },
        )
        device_id = require_str(f"{name}.device_id", device["device_id"])
        if device_id in seen_devices:
            raise S12Error(f"{name}.device_id: duplicate {device_id!r}")
        seen_devices.add(device_id)
        require_int(f"{name}.boot_epoch", device["boot_epoch"], 1)
        require_int(f"{name}.capacity_bytes", device["capacity_bytes"], 1)
        if require_int(f"{name}.activation_slots", device["activation_slots"], 1) != 1:
            raise S12Error(f"{name}.activation_slots: V2a requires exactly one")
        if require_int(f"{name}.prefetch_queue_limit", device["prefetch_queue_limit"], 1) != 1:
            raise S12Error(f"{name}.prefetch_queue_limit: V2a requires exactly one")
        _require_unique_strings(
            f"{name}.static_islands", device["static_islands"], nonempty=False
        )
        _require_unique_strings(
            f"{name}.dynamic_initial_islands",
            device["dynamic_initial_islands"],
            nonempty=False,
        )
    if len(devices) != 2:
        raise S12Error("config.devices: V2a requires exactly two devices")

    scheduler = require_exact_keys(
        "config.scheduler",
        config["scheduler"],
        {
            "min_prefetch_score_us",
            "prefetch_min_observations",
            "replicate_min_queue",
            "reuse_horizon_requests",
        },
    )
    require_int("config.scheduler.min_prefetch_score_us", scheduler["min_prefetch_score_us"])
    require_int("config.scheduler.prefetch_min_observations", scheduler["prefetch_min_observations"], 1)
    require_int("config.scheduler.replicate_min_queue", scheduler["replicate_min_queue"], 1)
    require_int("config.scheduler.reuse_horizon_requests", scheduler["reuse_horizon_requests"], 1)
    return config


def validate_trace(raw_records: list[Any], catalog: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    previous_arrival = -1
    for index, raw in enumerate(raw_records):
        name = f"trace[{index}]"
        record = require_exact_keys(
            name,
            raw,
            {
                "event_id",
                "arrival_us",
                "deadline_us",
                "priority_class",
                "model_id",
                "island_id",
                "provenance",
            },
        )
        event_id = require_str(f"{name}.event_id", record["event_id"])
        if event_id in seen_ids:
            raise S12Error(f"{name}.event_id: duplicate {event_id!r}")
        seen_ids.add(event_id)
        arrival = require_int(f"{name}.arrival_us", record["arrival_us"])
        deadline = require_int(f"{name}.deadline_us", record["deadline_us"], arrival)
        require_int(f"{name}.priority_class", record["priority_class"])
        model_id = require_str(f"{name}.model_id", record["model_id"])
        island_id = require_str(f"{name}.island_id", record["island_id"])
        require_str(f"{name}.provenance", record["provenance"], {"semi_synthetic"})
        if arrival < previous_arrival:
            raise S12Error(f"{name}.arrival_us: trace must be nondecreasing")
        previous_arrival = arrival
        if island_id not in catalog:
            raise S12Error(f"{name}.island_id: unknown island {island_id!r}")
        if catalog[island_id]["model_id"] != model_id:
            raise S12Error(f"{name}: model does not own island")
        records.append(record)
    if not records:
        raise S12Error("trace: expected at least one request")
    return records


def build_catalog(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for raw in profile["islands"]:
        island = dict(raw)
        island["routes"] = {route["device_id"]: dict(route) for route in raw["device_routes"]}
        catalog[island["island_id"]] = island
    return catalog


def validate_cross_inputs(config: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> None:
    device_ids = {device["device_id"] for device in config["devices"]}
    for island_id, island in catalog.items():
        unknown = set(island["routes"]) - device_ids
        if unknown:
            raise S12Error(f"profile island {island_id}: routes reference unknown devices {sorted(unknown)}")
    for device in config["devices"]:
        for key in ("static_islands", "dynamic_initial_islands"):
            total = 0
            for island_id in device[key]:
                if island_id not in catalog:
                    raise S12Error(f"config device {device['device_id']}: unknown island {island_id!r}")
                if device["device_id"] not in catalog[island_id]["routes"]:
                    raise S12Error(
                        f"config device {device['device_id']}: island {island_id!r} is incompatible"
                    )
                total += catalog[island_id]["weight_bytes"]
            if total > device["capacity_bytes"]:
                raise S12Error(f"config device {device['device_id']}: initial residency exceeds capacity")


@dataclass(frozen=True)
class ReplicaView:
    device_id: str
    island_id: str
    status: str
    pins: int
    generation: int
    boot_epoch: int
    status_seq: int
    weight_identity: str = ""
    model_id: str = ""
    backend: str = ""


@dataclass(frozen=True)
class DeviceView:
    device_id: str
    boot_epoch: int
    capacity_bytes: int
    used_bytes: int
    prefetch_active: int
    pending_island: str | None
    replicas: tuple[ReplicaView, ...]


@dataclass(frozen=True)
class SlowSnapshot:
    now_us: int
    queued_by_island: tuple[tuple[str, int], ...]
    observed_by_island: tuple[tuple[str, int], ...]
    devices: tuple[DeviceView, ...]


def plan_slow(
    snapshot: SlowSnapshot,
    catalog: dict[str, dict[str, Any]],
    scheduler: dict[str, int],
) -> list[dict[str, Any]]:
    """Return causal placement intents from an immutable present-state view."""
    queued = dict(snapshot.queued_by_island)
    observed = dict(snapshot.observed_by_island)
    all_replicas = [replica for device in snapshot.devices for replica in device.replicas]
    keep_options: list[dict[str, Any]] = []
    placement_options: list[dict[str, Any]] = []

    for island_id in sorted(catalog):
        demand = queued.get(island_id, 0)
        seen = observed.get(island_id, 0)
        if seen == 0 or demand == 0:
            continue
        replicas = [replica for replica in all_replicas if replica.island_id == island_id]
        usable = [replica for replica in replicas if replica.status in {"READY", "LEASED"}]
        in_progress = [
            replica
            for replica in replicas
            if replica.status in {"RECEIVING", "VERIFYING", "PREPARING", "DRAINING", "EVICTING"}
        ]
        pending = any(device.pending_island == island_id for device in snapshot.devices)

        for replica in sorted(usable, key=lambda item: item.device_id):
            if demand:
                keep_options.append(
                    {
                        "action": "KEEP",
                        "device_id": replica.device_id,
                        "expected_boot_epoch": replica.boot_epoch,
                        "expected_replica_generation": replica.generation,
                        "expected_replica_status_seq": replica.status_seq,
                        "expected_victim_generation": None,
                        "expected_victim_status_seq": None,
                        "island_id": island_id,
                        "backend": replica.backend,
                        "model_id": replica.model_id,
                        "score_us": 0,
                        "victim_island_id": None,
                        "weight_identity": replica.weight_identity,
                    }
                )

        action_kind: str | None = None
        if not replicas and not pending and seen >= scheduler["prefetch_min_observations"]:
            action_kind = "PREFETCH"
        elif (
            len(usable) == 1
            and not in_progress
            and not pending
            and demand >= scheduler["replicate_min_queue"]
        ):
            action_kind = "REPLICATE"
        if action_kind is None:
            continue

        for device in snapshot.devices:
            if device.device_id not in catalog[island_id]["routes"]:
                continue
            if any(replica.island_id == island_id for replica in device.replicas):
                continue
            if device.pending_island is not None:
                continue
            if device.prefetch_active:
                continue
            weight_bytes = catalog[island_id]["weight_bytes"]
            victim: ReplicaView | None = None
            if device.used_bytes + weight_bytes > device.capacity_bytes:
                victims = []
                for replica in device.replicas:
                    other_copy = any(
                        candidate.island_id == replica.island_id
                        and candidate.device_id != device.device_id
                        and candidate.status in {"READY", "LEASED"}
                        for candidate in all_replicas
                    )
                    if queued.get(replica.island_id, 0) == 0 or other_copy:
                        victims.append(replica)
                if not victims:
                    continue
                victim = sorted(victims, key=lambda item: (item.pins != 0, item.island_id))[0]
                if device.used_bytes - catalog[victim.island_id]["weight_bytes"] + weight_bytes > device.capacity_bytes:
                    continue
            route = catalog[island_id]["routes"][device.device_id]
            saving = catalog[island_id]["synthetic_server_full_us"] - (
                route["synthetic_phone_us"] + catalog[island_id]["synthetic_server_tail_us"]
            )
            reuse = min(demand, scheduler["reuse_horizon_requests"])
            provision = route["transfer_us"] + route["verify_us"] + route["prepare_us"]
            score = reuse * saving - provision
            if score < -MAX_SAFE_INT or score > MAX_SAFE_INT:
                raise S12Error("scheduler score exceeds canonical integer range")
            if score >= scheduler["min_prefetch_score_us"]:
                placement_options.append(
                {
                    "action": (
                        "DEFERRED_EVICT_PREFETCH" if victim is not None else action_kind
                    ),
                    "device_id": device.device_id,
                    "expected_boot_epoch": device.boot_epoch,
                    "expected_replica_generation": None,
                    "expected_replica_status_seq": None,
                    "expected_victim_generation": (
                        victim.generation if victim is not None else None
                    ),
                    "expected_victim_status_seq": (
                        victim.status_seq if victim is not None else None
                    ),
                    "island_id": island_id,
                    "backend": catalog[island_id]["routes"][device.device_id]["backend"],
                    "model_id": catalog[island_id]["model_id"],
                    "score_us": score,
                    "victim_island_id": victim.island_id if victim is not None else None,
                    "weight_identity": catalog[island_id]["weight_identity"],
                }
            )
    usable_counts: dict[str, int] = {}
    for replica in all_replicas:
        if replica.status in {"READY", "LEASED"}:
            usable_counts[replica.island_id] = usable_counts.get(replica.island_id, 0) + 1
    best: tuple[dict[str, Any], ...] = ()
    best_objective = (0, 0)
    best_signature: tuple[tuple[str, str, str], ...] = ()
    max_actions = min(len(snapshot.devices), len(placement_options))
    for count in range(max_actions + 1):
        for candidate in itertools.combinations(placement_options, count):
            if len({item["device_id"] for item in candidate}) != count:
                continue
            if len({item["island_id"] for item in candidate}) != count:
                continue
            victim_counts: dict[str, int] = {}
            for item in candidate:
                victim_id = item["victim_island_id"]
                if victim_id is not None:
                    victim_counts[victim_id] = victim_counts.get(victim_id, 0) + 1
            if any(
                queued.get(victim_id, 0)
                and usable_counts.get(victim_id, 0) - victim_count < 1
                for victim_id, victim_count in victim_counts.items()
            ):
                continue
            objective = (sum(item["score_us"] for item in candidate), count)
            signature = tuple(
                sorted(
                    (
                        item["island_id"],
                        item["device_id"],
                        item["victim_island_id"] or "",
                    )
                    for item in candidate
                )
            )
            if objective > best_objective or (
                objective == best_objective and (not best or signature < best_signature)
            ):
                best = candidate
                best_objective = objective
                best_signature = signature
    selected = sorted(best, key=lambda item: (item["device_id"], item["island_id"]))
    selected_victim_replicas = {
        (option["device_id"], option["victim_island_id"])
        for option in selected
        if option["victim_island_id"] is not None
    }
    keeps = [
        option
        for option in keep_options
        if (option["device_id"], option["island_id"]) not in selected_victim_replicas
    ]
    return keeps + selected


@dataclass
class RequestState:
    request_id: str
    arrival_us: int
    deadline_us: int
    priority_class: int
    model_id: str
    island_id: str
    terminal: str | None = None
    route: str | None = None
    device_id: str | None = None
    start_us: int | None = None
    phone_finish_us: int | None = None
    finish_us: int | None = None
    phase: str = "future"


@dataclass
class Replica:
    device_id: str
    island_id: str
    generation: int
    boot_epoch: int
    status: str
    status_seq: int = 1
    pins: int = 0
    ever_dispatched: bool = False
    completed_dispatches: int = 0
    transferred: bool = False
    weight_identity: str = ""
    model_id: str = ""
    backend: str = ""

    def transition(self, target: str) -> None:
        if target not in REPLICA_STATES:
            raise AssertionError(f"invalid replica state {target}")
        if target not in LEGAL_REPLICA_TRANSITIONS[self.status]:
            raise AssertionError(f"invalid replica transition {self.status}->{target}")
        self.status = target
        self.status_seq = _safe_add("replica.status_seq", self.status_seq, 1)


@dataclass
class DeviceState:
    device_id: str
    boot_epoch: int
    capacity_bytes: int
    activation_slots: int
    prefetch_queue_limit: int
    replicas: dict[str, Replica] = field(default_factory=dict)
    pending_intent: dict[str, Any] | None = None
    used_bytes: int = 0
    peak_bytes: int = 0
    activation_used: int = 0
    peak_activation_used: int = 0
    compute_request_id: str | None = None
    generation_counter: int = 0
    prefetch_active: int = 0
    peak_prefetch_active: int = 0
    compute_intervals: list[tuple[int, int, str]] = field(default_factory=list)


def _overlap_us(
    left: list[tuple[int, int, str]],
    right: list[tuple[int, int, str]],
) -> int:
    total = 0
    for left_start, left_finish, _ in left:
        for right_start, right_finish, _ in right:
            total += max(0, min(left_finish, right_finish) - max(left_start, right_start))
    return total


class Simulator:
    def __init__(
        self,
        policy: str,
        config: dict[str, Any],
        catalog: dict[str, dict[str, Any]],
        trace: list[dict[str, Any]],
    ) -> None:
        self.policy = require_str("policy", policy, set(POLICIES))
        self.config = config
        self.catalog = catalog
        self.requests = [
            RequestState(
                request_id=record["event_id"],
                arrival_us=record["arrival_us"],
                deadline_us=record["deadline_us"],
                priority_class=record["priority_class"],
                model_id=record["model_id"],
                island_id=record["island_id"],
            )
            for record in trace
        ]
        self.request_by_id = {request.request_id: request for request in self.requests}
        self.devices = {
            raw["device_id"]: DeviceState(
                device_id=raw["device_id"],
                boot_epoch=raw["boot_epoch"],
                capacity_bytes=raw["capacity_bytes"],
                activation_slots=raw["activation_slots"],
                prefetch_queue_limit=raw["prefetch_queue_limit"],
            )
            for raw in config["devices"]
        }
        self.device_configs = {raw["device_id"]: raw for raw in config["devices"]}
        self.events: list[tuple[int, int, int, str, dict[str, Any]]] = []
        self.event_sequence = 0
        self.now_us = 0
        self.queue: list[str] = []
        self.tail_queue: list[tuple[int, str]] = []
        self.server_request_id: str | None = None
        self.server_phase: str | None = None
        self.server_intervals: list[tuple[int, int, str, str]] = []
        self.observed: dict[str, int] = {island_id: 0 for island_id in catalog}
        self.slow_actions: list[dict[str, Any]] = []
        self.fast_actions: list[dict[str, Any]] = []
        self.keep_logged: set[tuple[str, str, int]] = set()
        self.transfer_bytes_completed = 0
        self.evicted_bytes = 0
        self.cancelled_prefetch_reservation_bytes = 0
        self.stale_residency_events = 0
        self.replica_history: list[Replica] = []
        self.max_queue_depth = 0
        self.initial_residency_bytes = 0
        self._initialize_replicas()
        for request in self.requests:
            self._push(request.arrival_us, 2, "arrival", {"request_id": request.request_id})

    def _initialize_replicas(self) -> None:
        for device_id in sorted(self.devices):
            device = self.devices[device_id]
            raw = self.device_configs[device_id]
            if self.policy in {"static_two_phone", "fixed_static_phone"}:
                initial = raw["static_islands"]
            elif self.policy == "dynamic_two_level":
                initial = raw["dynamic_initial_islands"]
            else:
                initial = []
            for island_id in initial:
                device.generation_counter = _safe_add(
                    "device.generation_counter", device.generation_counter, 1
                )
                replica = Replica(
                    device_id=device_id,
                    island_id=island_id,
                    generation=device.generation_counter,
                    boot_epoch=device.boot_epoch,
                    status="READY",
                    weight_identity=self.catalog[island_id]["weight_identity"],
                    model_id=self.catalog[island_id]["model_id"],
                    backend=self.catalog[island_id]["routes"][device_id]["backend"],
                )
                device.replicas[island_id] = replica
                self.replica_history.append(replica)
                device.used_bytes = _safe_add(
                    "device.used_bytes",
                    device.used_bytes,
                    self.catalog[island_id]["weight_bytes"],
                )
                device.peak_bytes = max(device.peak_bytes, device.used_bytes)
                self.initial_residency_bytes = _safe_add(
                    "initial_residency_bytes",
                    self.initial_residency_bytes,
                    self.catalog[island_id]["weight_bytes"],
                )

    def _push(self, at_us: int, priority: int, kind: str, payload: dict[str, Any]) -> None:
        require_int("event.at_us", at_us)
        self.event_sequence = _safe_add("event_sequence", self.event_sequence, 1)
        heapq.heappush(self.events, (at_us, priority, self.event_sequence, kind, payload))

    def _snapshot(self) -> SlowSnapshot:
        queued_counts = {island_id: 0 for island_id in self.catalog}
        for request_id in self.queue:
            request = self.request_by_id[request_id]
            if request.terminal is None and request.phase == "queued":
                queued_counts[request.island_id] += 1
        devices = []
        for device_id in sorted(self.devices):
            device = self.devices[device_id]
            replicas = tuple(
                ReplicaView(
                    device_id=device_id,
                    island_id=replica.island_id,
                    status=replica.status,
                    pins=replica.pins,
                    generation=replica.generation,
                    boot_epoch=replica.boot_epoch,
                    status_seq=replica.status_seq,
                    weight_identity=replica.weight_identity,
                    model_id=replica.model_id,
                    backend=replica.backend,
                )
                for replica in sorted(device.replicas.values(), key=lambda item: item.island_id)
            )
            devices.append(
                DeviceView(
                    device_id=device_id,
                    boot_epoch=device.boot_epoch,
                    capacity_bytes=device.capacity_bytes,
                    used_bytes=device.used_bytes,
                    prefetch_active=device.prefetch_active,
                    pending_island=(
                        device.pending_intent["island_id"] if device.pending_intent is not None else None
                    ),
                    replicas=replicas,
                )
            )
        return SlowSnapshot(
            now_us=self.now_us,
            queued_by_island=tuple(sorted(queued_counts.items())),
            observed_by_island=tuple(sorted(self.observed.items())),
            devices=tuple(devices),
        )

    def _record_slow(self, action: dict[str, Any], applied: bool, reason: str) -> None:
        self.slow_actions.append(
            {
                "action": action["action"],
                "applied": applied,
                "backend": action["backend"],
                "device_id": action["device_id"],
                "expected_boot_epoch": action["expected_boot_epoch"],
                "expected_replica_generation": action["expected_replica_generation"],
                "expected_replica_status_seq": action["expected_replica_status_seq"],
                "expected_victim_generation": action["expected_victim_generation"],
                "expected_victim_status_seq": action["expected_victim_status_seq"],
                "island_id": action["island_id"],
                "model_id": action["model_id"],
                "reason": reason,
                "score_us": action["score_us"],
                "time_us": self.now_us,
                "victim_island_id": action["victim_island_id"],
                "weight_identity": action["weight_identity"],
            }
        )

    def _run_slow_loop(self) -> None:
        if self.policy != "dynamic_two_level":
            return
        actions = plan_slow(self._snapshot(), self.catalog, self.config["scheduler"])
        for action in actions:
            if action["action"] == "KEEP":
                self._apply_keep(action)
                continue
            self._apply_residency_intent(action)

    def _apply_keep(self, action: dict[str, Any]) -> None:
        device = self.devices[action["device_id"]]
        replica = device.replicas.get(action["island_id"])
        island = self.catalog[action["island_id"]]
        route = island["routes"][device.device_id]
        if device.boot_epoch != action["expected_boot_epoch"]:
            self._record_slow(action, False, "stale_boot_epoch")
            return
        if (
            replica is None
            or replica.generation != action["expected_replica_generation"]
            or replica.status_seq != action["expected_replica_status_seq"]
            or replica.status not in {"READY", "LEASED"}
            or replica.weight_identity != action["weight_identity"]
            or replica.model_id != action["model_id"]
            or replica.backend != action["backend"]
            or action["weight_identity"] != island["weight_identity"]
            or action["model_id"] != island["model_id"]
            or action["backend"] != route["backend"]
        ):
            self._record_slow(action, False, "stale_replica_epoch")
            return
        key = (
            action["device_id"],
            action["island_id"],
            action["expected_replica_generation"],
        )
        if key not in self.keep_logged:
            self.keep_logged.add(key)
            self._record_slow(action, True, "coherent_current_replica")

    def _apply_residency_intent(self, action: dict[str, Any]) -> None:
        device = self.devices[action["device_id"]]
        island_id = action["island_id"]
        if device.boot_epoch != action["expected_boot_epoch"]:
            self._record_slow(action, False, "stale_boot_epoch")
            return
        if (
            action["weight_identity"] != self.catalog[island_id]["weight_identity"]
            or action["model_id"] != self.catalog[island_id]["model_id"]
            or action["backend"] != self.catalog[island_id]["routes"][device.device_id]["backend"]
        ):
            self._record_slow(action, False, "stale_content_identity")
            return
        if (
            island_id in device.replicas
            or device.pending_intent is not None
            or device.prefetch_active >= device.prefetch_queue_limit
        ):
            self._record_slow(action, False, "duplicate_or_busy")
            return
        victim_id = action["victim_island_id"]
        if victim_id is None:
            self._record_slow(action, True, "capacity_reserved")
            self._begin_prefetch(device, island_id)
            return
        victim = device.replicas.get(victim_id)
        if victim is None or victim.status not in {"READY", "LEASED"}:
            self._record_slow(action, False, "victim_not_evictable")
            return
        if (
            victim.generation != action["expected_victim_generation"]
            or victim.status_seq != action["expected_victim_status_seq"]
        ):
            self._record_slow(action, False, "stale_victim_epoch")
            return
        victim_has_demand = any(
            request.terminal is None
            and request.phase == "queued"
            and request.island_id == victim.island_id
            for request in self.requests
        )
        other_usable_copy = any(
            other.device_id != device.device_id
            and other.island_id == victim.island_id
            and other.status in {"READY", "LEASED"}
            for other_device in self.devices.values()
            for other in other_device.replicas.values()
        )
        if victim_has_demand and not other_usable_copy:
            self._record_slow(action, False, "last_demanded_replica")
            return
        device.pending_intent = dict(action)
        if victim.pins:
            victim.transition("DRAINING")
            self._record_slow(action, True, "victim_pinned_drain_started")
        else:
            self._record_slow(action, True, "victim_unpinned_evicted")
            self._finish_eviction_and_prefetch(device, victim)

    def _begin_prefetch(self, device: DeviceState, island_id: str) -> None:
        island = self.catalog[island_id]
        weight_bytes = island["weight_bytes"]
        if device.prefetch_active >= device.prefetch_queue_limit:
            raise AssertionError("prefetch queue overflow")
        if device.used_bytes + weight_bytes > device.capacity_bytes:
            raise AssertionError("prefetch exceeds device capacity")
        device.generation_counter = _safe_add(
            "device.generation_counter", device.generation_counter, 1
        )
        replica = Replica(
            device_id=device.device_id,
            island_id=island_id,
            generation=device.generation_counter,
            boot_epoch=device.boot_epoch,
            status="RECEIVING",
            transferred=True,
            weight_identity=island["weight_identity"],
            model_id=island["model_id"],
            backend=island["routes"][device.device_id]["backend"],
        )
        device.replicas[island_id] = replica
        self.replica_history.append(replica)
        device.used_bytes = _safe_add("device.used_bytes", device.used_bytes, weight_bytes)
        device.peak_bytes = max(device.peak_bytes, device.used_bytes)
        device.prefetch_active += 1
        device.peak_prefetch_active = max(device.peak_prefetch_active, device.prefetch_active)
        route = island["routes"][device.device_id]
        self._push(
            self.now_us + route["transfer_us"],
            1,
            "residency_transition",
            {
                "device_id": device.device_id,
                "generation": replica.generation,
                "island_id": island_id,
                "boot_epoch": replica.boot_epoch,
                "expected_status_seq": replica.status_seq,
                "target": "VERIFYING",
            },
        )

    def _finish_eviction_and_prefetch(self, device: DeviceState, victim: Replica) -> None:
        if victim.pins:
            raise AssertionError("attempted to evict pinned replica")
        victim.transition("EVICTING")
        weight_bytes = self.catalog[victim.island_id]["weight_bytes"]
        device.used_bytes -= weight_bytes
        self.evicted_bytes = _safe_add("evicted_bytes", self.evicted_bytes, weight_bytes)
        del device.replicas[victim.island_id]
        pending = device.pending_intent
        if pending is None:
            return
        device.pending_intent = None
        self._begin_prefetch(device, pending["island_id"])

    def _handle_residency_transition(self, payload: dict[str, Any]) -> None:
        device = self.devices[payload["device_id"]]
        replica = device.replicas.get(payload["island_id"])
        if (
            replica is None
            or replica.generation != payload["generation"]
            or replica.boot_epoch != payload["boot_epoch"]
            or replica.status_seq != payload["expected_status_seq"]
        ):
            self.stale_residency_events = _safe_add(
                "stale_residency_events", self.stale_residency_events, 1
            )
            return
        target = payload["target"]
        route = self.catalog[replica.island_id]["routes"][device.device_id]
        if target == "VERIFYING" and replica.status == "RECEIVING":
            replica.transition("VERIFYING")
            self._push(
                self.now_us + route["verify_us"],
                1,
                "residency_transition",
                {
                    **payload,
                    "expected_status_seq": replica.status_seq,
                    "target": "PREPARING",
                },
            )
        elif target == "PREPARING" and replica.status == "VERIFYING":
            replica.transition("PREPARING")
            self._push(
                self.now_us + route["prepare_us"],
                1,
                "residency_transition",
                {
                    **payload,
                    "expected_status_seq": replica.status_seq,
                    "target": "READY",
                },
            )
        elif target == "READY" and replica.status == "PREPARING":
            replica.transition("READY")
            device.prefetch_active -= 1
            self.transfer_bytes_completed = _safe_add(
                "transfer_bytes_completed",
                self.transfer_bytes_completed,
                self.catalog[replica.island_id]["weight_bytes"],
            )
        else:
            raise AssertionError(
                f"invalid residency transition {replica.status}->{target} for {replica.island_id}"
            )

    def _handle_arrival(self, request: RequestState) -> None:
        if request.phase != "future":
            raise AssertionError("duplicate arrival")
        self.observed[request.island_id] = _safe_add(
            "observed_arrivals", self.observed[request.island_id], 1
        )
        active_queue = sum(
            1
            for request_id in self.queue
            if self.request_by_id[request_id].terminal is None
            and self.request_by_id[request_id].phase == "queued"
        )
        if active_queue >= self.config["queue_limit"]:
            request.phase = "terminal"
            request.terminal = "rejected_queue_full"
            request.finish_us = self.now_us
            return
        request.phase = "queued"
        self.queue.append(request.request_id)
        self.max_queue_depth = max(self.max_queue_depth, active_queue + 1)

    def _dispatch_phones(self) -> bool:
        progress = False
        for device_id in sorted(self.devices):
            device = self.devices[device_id]
            if device.compute_request_id is not None or device.activation_used >= device.activation_slots:
                continue
            candidate: RequestState | None = None
            replica: Replica | None = None
            for request_id in self.queue:
                request = self.request_by_id[request_id]
                if request.terminal is not None or request.phase != "queued":
                    continue
                current = device.replicas.get(request.island_id)
                if current is not None and current.status == "READY":
                    island = self.catalog[request.island_id]
                    route = island["routes"][device_id]
                    if (
                        current.boot_epoch != device.boot_epoch
                        or current.generation > device.generation_counter
                        or current.weight_identity != island["weight_identity"]
                        or current.model_id != request.model_id
                        or current.model_id != island["model_id"]
                        or current.backend != route["backend"]
                    ):
                        raise AssertionError("replica identity mismatch before dispatch")
                    candidate = request
                    replica = current
                    break
            if candidate is None or replica is None:
                continue
            candidate.phase = "phone_compute"
            candidate.route = "phone"
            candidate.device_id = device_id
            candidate.start_us = self.now_us
            replica.transition("LEASED")
            replica.pins += 1
            replica.ever_dispatched = True
            device.activation_used += 1
            device.peak_activation_used = max(device.peak_activation_used, device.activation_used)
            device.compute_request_id = candidate.request_id
            duration = self.catalog[candidate.island_id]["routes"][device_id]["synthetic_phone_us"]
            finish = self.now_us + duration
            device.compute_intervals.append((self.now_us, finish, candidate.request_id))
            self.fast_actions.append(
                {
                    "action": "DISPATCH_PHONE",
                    "backend": replica.backend,
                    "boot_epoch": replica.boot_epoch,
                    "device_id": device_id,
                    "generation": replica.generation,
                    "island_id": candidate.island_id,
                    "model_id": replica.model_id,
                    "request_ids": [candidate.request_id],
                    "batch_size": 1,
                    "start_us": self.now_us,
                    "status_before": "READY",
                    "status_seq_before": replica.status_seq - 1,
                    "weight_identity": replica.weight_identity,
                }
            )
            self._push(
                finish,
                0,
                "phone_done",
                {"device_id": device_id, "request_id": candidate.request_id},
            )
            progress = True
        return progress

    def _dispatch_server(self) -> bool:
        if self.policy == "fixed_static_phone":
            return False
        if self.server_request_id is not None:
            return False
        request: RequestState | None = None
        phase: str
        if self.tail_queue:
            self.tail_queue.sort(key=lambda item: (item[0], item[1]))
            _, request_id = self.tail_queue.pop(0)
            request = self.request_by_id[request_id]
            phase = "tail"
            duration = self.catalog[request.island_id]["synthetic_server_tail_us"]
            request.phase = "server_tail"
            action = "BATCH_TAIL"
        else:
            candidates = [
                self.request_by_id[request_id]
                for request_id in self.queue
                if self.request_by_id[request_id].terminal is None
                and self.request_by_id[request_id].phase == "queued"
            ]
            if not candidates:
                return False
            request = sorted(candidates, key=lambda item: (item.arrival_us, item.request_id))[0]
            phase = "full"
            duration = self.catalog[request.island_id]["synthetic_server_full_us"]
            request.phase = "server_full"
            request.route = "server"
            request.start_us = self.now_us
            action = "SERVER_FALLBACK"
        self.server_request_id = request.request_id
        self.server_phase = phase
        finish = self.now_us + duration
        self.server_intervals.append((self.now_us, finish, request.request_id, phase))
        self.fast_actions.append(
            {
                "action": action,
                "batch_size": 1,
                "device_id": request.device_id if phase == "tail" else None,
                "island_id": request.island_id,
                "model_id": request.model_id,
                "request_ids": [request.request_id],
                "start_us": self.now_us,
            }
        )
        self._push(finish, 0, "server_done", {"phase": phase, "request_id": request.request_id})
        return True

    def _handle_phone_done(self, payload: dict[str, Any]) -> None:
        request = self.request_by_id[payload["request_id"]]
        device = self.devices[payload["device_id"]]
        if device.compute_request_id != request.request_id or request.phase != "phone_compute":
            raise AssertionError("phone completion ownership mismatch")
        device.compute_request_id = None
        request.phone_finish_us = self.now_us
        request.phase = "tail_wait"
        self.tail_queue.append((self.now_us, request.request_id))

    def _handle_server_done(self, payload: dict[str, Any]) -> None:
        request = self.request_by_id[payload["request_id"]]
        if self.server_request_id != request.request_id or self.server_phase != payload["phase"]:
            raise AssertionError("server completion ownership mismatch")
        self.server_request_id = None
        self.server_phase = None
        request.finish_us = self.now_us
        tardy = self.now_us > request.deadline_us
        if payload["phase"] == "full":
            request.terminal = "tardy_server" if tardy else "completed_server"
        else:
            request.terminal = "tardy_phone" if tardy else "completed_phone"
            if request.device_id is None:
                raise AssertionError("phone completion has no device")
            replica = self.devices[request.device_id].replicas.get(request.island_id)
            if replica is None:
                raise AssertionError("phone completion lost its replica")
            replica.completed_dispatches = _safe_add(
                "replica.completed_dispatches", replica.completed_dispatches, 1
            )
            self._release_phone_lease(request)
        request.phase = "terminal"

    def _release_phone_lease(self, request: RequestState) -> None:
        if request.device_id is None:
            raise AssertionError("phone request has no device")
        device = self.devices[request.device_id]
        replica = device.replicas.get(request.island_id)
        if replica is None or replica.pins != 1 or device.activation_used != 1:
            raise AssertionError("phone lease accounting mismatch")
        replica.pins -= 1
        device.activation_used -= 1
        if replica.status == "LEASED":
            replica.transition("READY")
        elif replica.status == "DRAINING":
            self._finish_eviction_and_prefetch(device, replica)
        else:
            raise AssertionError(f"lease released from invalid state {replica.status}")

    def _process_event(self, kind: str, payload: dict[str, Any]) -> None:
        if kind == "arrival":
            self._handle_arrival(self.request_by_id[payload["request_id"]])
        elif kind == "residency_transition":
            self._handle_residency_transition(payload)
        elif kind == "phone_done":
            self._handle_phone_done(payload)
        elif kind == "server_done":
            self._handle_server_done(payload)
        else:
            raise AssertionError(f"unknown event {kind}")

    def _assert_invariants(self) -> None:
        if self.server_request_id is None and self.server_phase is not None:
            raise AssertionError("orphan server phase")
        queued = 0
        for request_id in self.queue:
            request = self.request_by_id[request_id]
            if request.terminal is None and request.phase == "queued":
                queued += 1
        if queued > self.config["queue_limit"]:
            raise AssertionError("virtual queue overflow")
        for device in self.devices.values():
            expected = sum(self.catalog[replica.island_id]["weight_bytes"] for replica in device.replicas.values())
            if expected != device.used_bytes or not (0 <= device.used_bytes <= device.capacity_bytes):
                raise AssertionError("device LPDDR ledger mismatch")
            pins = sum(replica.pins for replica in device.replicas.values())
            if pins != device.activation_used:
                raise AssertionError("pin/activation ledger mismatch")
            if not (0 <= device.activation_used <= device.activation_slots):
                raise AssertionError("activation slot overflow")
            if not (0 <= device.prefetch_active <= device.prefetch_queue_limit):
                raise AssertionError("prefetch queue overflow")
            for replica in device.replicas.values():
                if replica.boot_epoch != device.boot_epoch or replica.generation > device.generation_counter:
                    raise AssertionError("stale replica identity")
                island = self.catalog[replica.island_id]
                if (
                    replica.weight_identity != island["weight_identity"]
                    or replica.model_id != island["model_id"]
                    or replica.backend != island["routes"][device.device_id]["backend"]
                ):
                    raise AssertionError("replica content identity mismatch")
                if replica.pins and replica.status not in {"LEASED", "DRAINING"}:
                    raise AssertionError("pinned replica has invalid state")
            intervals = sorted(device.compute_intervals)
            for previous, current in zip(intervals, intervals[1:]):
                if previous[1] > current[0]:
                    raise AssertionError("same-device compute overlap")
        server = sorted(self.server_intervals)
        for previous, current in zip(server, server[1:]):
            if previous[1] > current[0]:
                raise AssertionError("server lane overlap")

    def _terminalize_horizon(self) -> None:
        horizon = self.config["horizon_us"]
        for request in self.requests:
            if request.terminal is not None:
                continue
            if request.route == "phone" and request.device_id is not None:
                device = self.devices[request.device_id]
                replica = device.replicas.get(request.island_id)
                if replica is not None and replica.pins:
                    replica.pins -= 1
                    device.activation_used -= 1
                    if replica.status == "LEASED":
                        replica.transition("READY")
                if device.compute_request_id == request.request_id:
                    device.compute_request_id = None
            request.terminal = "timed_out"
            request.phase = "terminal"
            request.finish_us = horizon
        self.server_request_id = None
        self.server_phase = None
        self.tail_queue.clear()
        for device in self.devices.values():
            device.pending_intent = None
            for island_id, replica in list(device.replicas.items()):
                weight_bytes = self.catalog[island_id]["weight_bytes"]
                if replica.status == "DRAINING" and replica.pins == 0:
                    replica.transition("EVICTING")
                    device.used_bytes -= weight_bytes
                    self.evicted_bytes = _safe_add("evicted_bytes", self.evicted_bytes, weight_bytes)
                    del device.replicas[island_id]
                elif replica.status in {"RECEIVING", "VERIFYING", "PREPARING"}:
                    replica.transition("EVICTING")
                    device.used_bytes -= weight_bytes
                    self.cancelled_prefetch_reservation_bytes = _safe_add(
                        "cancelled_prefetch_reservation_bytes",
                        self.cancelled_prefetch_reservation_bytes,
                        weight_bytes,
                    )
                    del device.replicas[island_id]
            device.prefetch_active = 0
            device.compute_intervals = [
                (start, min(finish, horizon), request_id)
                for start, finish, request_id in device.compute_intervals
                if start < horizon
            ]
        self.server_intervals = [
            (start, min(finish, horizon), request_id, phase)
            for start, finish, request_id, phase in self.server_intervals
            if start < horizon
        ]

    def run(self) -> dict[str, Any]:
        horizon = self.config["horizon_us"]
        while self.events and self.events[0][0] <= horizon:
            self.now_us = self.events[0][0]
            current: list[tuple[int, int, int, str, dict[str, Any]]] = []
            while self.events and self.events[0][0] == self.now_us:
                current.append(heapq.heappop(self.events))
            for _, _, _, kind, payload in current:
                self._process_event(kind, payload)
            if self.now_us < horizon:
                self._run_slow_loop()
                self._dispatch_phones()
                self._dispatch_server()
            self._assert_invariants()
        self.now_us = horizon
        self._terminalize_horizon()
        self._assert_invariants()
        return self._result()

    def _result(self) -> dict[str, Any]:
        counts = {terminal: 0 for terminal in TERMINALS}
        outcomes = []
        for request in self.requests:
            if request.terminal not in counts:
                raise AssertionError("request missing terminal")
            counts[request.terminal] += 1
            outcomes.append(
                {
                    "arrival_us": request.arrival_us,
                    "deadline_us": request.deadline_us,
                    "device_id": request.device_id,
                    "finish_us": request.finish_us,
                    "island_id": request.island_id,
                    "model_id": request.model_id,
                    "priority_class": request.priority_class,
                    "request_id": request.request_id,
                    "route": request.route,
                    "start_us": request.start_us,
                    "terminal": request.terminal,
                }
            )
        if sum(counts.values()) != len(self.requests):
            raise AssertionError("terminal conservation failed")

        device_ledgers = []
        for device_id in sorted(self.devices):
            device = self.devices[device_id]
            final_residency = [
                {
                    "generation": replica.generation,
                    "backend": replica.backend,
                    "island_id": replica.island_id,
                    "model_id": replica.model_id,
                    "pins": replica.pins,
                    "status": replica.status,
                    "status_seq": replica.status_seq,
                    "weight_identity": replica.weight_identity,
                }
                for replica in sorted(device.replicas.values(), key=lambda item: item.island_id)
            ]
            device_ledgers.append(
                {
                    "activation_slots": device.activation_slots,
                    "activation_used_final": device.activation_used,
                    "capacity_bytes": device.capacity_bytes,
                    "compute_intervals": [
                        {"finish_us": finish, "request_id": request_id, "start_us": start}
                        for start, finish, request_id in device.compute_intervals
                    ],
                    "device_id": device_id,
                    "final_residency": final_residency,
                    "peak_activation_used": device.peak_activation_used,
                    "peak_prefetch_active": device.peak_prefetch_active,
                    "prefetch_active_final": device.prefetch_active,
                    "peak_resident_bytes": device.peak_bytes,
                    "resident_bytes_final": device.used_bytes,
                }
            )

        transferred = [replica for replica in self.replica_history if replica.transferred]
        useful_transfer_bytes = 0
        for replica in transferred:
            if replica.completed_dispatches:
                useful_transfer_bytes = _safe_add(
                    "useful_transfer_bytes",
                    useful_transfer_bytes,
                    self.catalog[replica.island_id]["weight_bytes"],
                )
        if useful_transfer_bytes > self.transfer_bytes_completed:
            raise AssertionError("useful transfer bytes exceed completed transfer bytes")
        ordered_devices = [self.devices[device_id] for device_id in sorted(self.devices)]
        overlap = _overlap_us(
            ordered_devices[0].compute_intervals,
            ordered_devices[1].compute_intervals,
        )
        result = {
            "cancelled_prefetch_reservation_bytes": self.cancelled_prefetch_reservation_bytes,
            "device_ledgers": device_ledgers,
            "energy": {"status": ENERGY_STATUS},
            "evicted_bytes": self.evicted_bytes,
            "fast_actions": self.fast_actions,
            "initial_residency_bytes": self.initial_residency_bytes,
            "max_queue_depth": self.max_queue_depth,
            "outcomes": outcomes,
            "policy": self.policy,
            "request_count": len(self.requests),
            "schema": "s12-two-level-policy-result-v1",
            "scope": SCOPE,
            "server_intervals": [
                {
                    "finish_us": finish,
                    "phase": phase,
                    "request_id": request_id,
                    "start_us": start,
                }
                for start, finish, request_id, phase in self.server_intervals
            ],
            "slow_actions": self.slow_actions,
            "stale_residency_events": self.stale_residency_events,
            "terminal_conservation": sum(counts.values()),
            "terminal_counts": counts,
            "transfer_bytes_completed": self.transfer_bytes_completed,
            "two_phone_compute_overlap_us": overlap,
            "useful_transfer_bytes": useful_transfer_bytes,
            "wasted_transfer_bytes": self.transfer_bytes_completed - useful_transfer_bytes,
        }
        result["result_digest"] = sha256_object(result)
        return result


def run_config(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config_bytes = read_bytes_once(config_file)
    config = validate_config(_decode_ascii_json(config_bytes, str(config_file)))
    profile_file = (config_file.parent / config["profile_path"]).resolve()
    trace_file = (config_file.parent / config["trace_path"]).resolve()
    profile_bytes = read_bytes_once(profile_file)
    profile = validate_profile(_decode_ascii_json(profile_bytes, str(profile_file)))
    catalog = build_catalog(profile)
    validate_cross_inputs(config, catalog)
    trace_bytes = read_bytes_once(trace_file)
    trace = validate_trace(parse_jsonl_bytes(trace_bytes, str(trace_file)), catalog)
    if trace[-1]["arrival_us"] > config["horizon_us"]:
        raise S12Error("trace: arrival exceeds replay horizon")

    results = [Simulator(policy, config, catalog, trace).run() for policy in config["policies"]]
    manifest = {
        "energy": {"status": ENERGY_STATUS},
        "inputs": {
            "config_path": str(config_file),
            "config_sha256": sha256_bytes(config_bytes),
            "config_semantic_sha256": sha256_object(
                {
                    key: value
                    for key, value in config.items()
                    if key not in {"profile_path", "trace_path"}
                }
            ),
            "profile_path": str(profile_file),
            "profile_sha256": sha256_bytes(profile_bytes),
            "trace_path": str(trace_file),
            "trace_sha256": sha256_bytes(trace_bytes),
        },
        "results": results,
        "schema": "s12-two-level-replay-v1",
        "scope": SCOPE,
    }
    digest_preimage = {
        "energy": manifest["energy"],
        "input_hashes": {
            "config_semantic_sha256": manifest["inputs"]["config_semantic_sha256"],
            "profile_sha256": manifest["inputs"]["profile_sha256"],
            "trace_sha256": manifest["inputs"]["trace_sha256"],
        },
        "results": manifest["results"],
        "schema": manifest["schema"],
        "scope": manifest["scope"],
    }
    manifest["deterministic_replay_sha256"] = sha256_object(digest_preimage)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        result = run_config(args.config)
        if args.output:
            write_canonical(args.output, result)
        else:
            print(canonical_json(result))
        return 0
    except (S12Error, AssertionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
