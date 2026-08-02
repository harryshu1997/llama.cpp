#!/usr/bin/env python3
"""Deterministic reference for the shared queue-aware switch policy."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from evidence_common import EvidenceError, require, require_int, require_string


@dataclass(frozen=True)
class Demand:
    request_id: str
    model_id: str
    arrival_order: int


class QueueAwarePolicy:
    def __init__(
            self,
            *,
            models: list[str],
            gpu_model: str,
            gpu_slots: int,
            executor_order: list[str],
            promotion_enabled: bool = True) -> None:
        require(len(models) >= 2 and len(set(models)) == len(models),
                "models: expected unique values")
        require(gpu_model in models, "gpu_model: unknown model")
        require_int(gpu_slots, "gpu_slots", 1)
        require(
            executor_order and len(set(executor_order)) == len(executor_order),
            "executor_order: expected unique values",
        )
        self.models = tuple(models)
        self.gpu_model = gpu_model
        self.gpu_slots = gpu_slots
        self.executor_order = tuple(executor_order)
        require(isinstance(promotion_enabled, bool),
                "promotion_enabled: expected bool")
        self.promotion_enabled = promotion_enabled
        self.queues = {model_id: deque() for model_id in self.models}
        self.requests: dict[str, dict[str, Any]] = {}
        self.executors: dict[str, dict[str, Any]] = {
            "GPU": {
                "capacity": gpu_slots,
                "credit": gpu_slots,
                "model_id": gpu_model,
                "ready": True,
            }
        }
        self.proposed_target: str | None = None
        self.transition_target: str | None = None
        self.switch_intents: deque[dict[str, Any]] = deque()
        self.last_switch_order = -1
        self.last_arrival_order = -1
        self.actions: list[dict[str, Any]] = []

    def _emit(self, kind: str, **fields: Any) -> dict[str, Any]:
        action = {"action_index": len(self.actions), "kind": kind, **fields}
        self.actions.append(action)
        return action

    def set_executor(
            self,
            executor_id: str,
            *,
            model_id: str,
            capacity: int,
            ready: bool) -> None:
        require_string(executor_id, "executor_id")
        require(model_id in self.models, "executor model: unknown")
        require_int(capacity, "executor capacity", 1)
        previous = self.executors.get(executor_id)
        in_use = 0 if previous is None else previous["capacity"] - previous["credit"]
        require(in_use == 0, "executor: cannot rebind with in-flight requests")
        self.executors[executor_id] = {
            "capacity": capacity,
            "credit": capacity if ready else 0,
            "model_id": model_id,
            "ready": ready,
        }
        self._emit(
            "EXECUTOR_READY" if ready else "EXECUTOR_UNREADY",
            executor_id=executor_id,
            model_id=model_id,
        )
        self.reconsider()

    def arrive(
            self,
            request_id: str,
            model_id: str,
            arrival_order: int) -> list[dict[str, Any]]:
        require_string(request_id, "request_id")
        require(request_id not in self.requests, "request: duplicate ID")
        require(model_id in self.models, "request: unknown model")
        require_int(arrival_order, "arrival_order")
        require(arrival_order > self.last_arrival_order,
                "request: arrival order must increase")
        self.last_arrival_order = arrival_order
        demand = Demand(request_id, model_id, arrival_order)
        self.requests[request_id] = {
            "demand": demand,
            "executor_id": None,
            "state": "QUEUED",
        }
        self.queues[model_id].append(request_id)
        self._emit(
            "REQUEST_QUEUED",
            arrival_order=arrival_order,
            model_id=model_id,
            request_id=request_id,
        )
        return self.reconsider()

    def _available_executor(self, model_id: str) -> str | None:
        ordered = list(self.executor_order)
        for executor_id in sorted(self.executors):
            if executor_id not in ordered:
                ordered.append(executor_id)
        for executor_id in ordered:
            record = self.executors.get(executor_id)
            if executor_id == "GPU" and self.transition_target is not None:
                continue
            if record is not None and record["ready"] \
                    and record["model_id"] == model_id \
                    and record["credit"] > 0:
                return executor_id
        return None

    def _queued_demands(self) -> list[Demand]:
        return sorted(
            (
                record["demand"]
                for record in self.requests.values()
                if record["state"] == "QUEUED"
            ),
            key=lambda item: item.arrival_order,
        )

    def _has_nonterminal_demand(self, model_id: str) -> bool:
        return any(
            record["demand"].model_id == model_id
            and record["state"] in {"QUEUED", "IN_FLIGHT"}
            for record in self.requests.values()
        )

    def _refresh_switch_proposal(
            self, automatic_target: str | None) -> None:
        if self.transition_target is not None or not self.promotion_enabled:
            return
        while self.switch_intents:
            intent = self.switch_intents[0]
            target = intent["target_model_id"]
            if target != self.gpu_model and self._has_nonterminal_demand(target):
                break
            obsolete = self.switch_intents.popleft()
            replacement = (
                self.switch_intents[0]["target_model_id"]
                if self.switch_intents else automatic_target
            )
            self._emit(
                "SWITCH_COALESCED",
                intent_id=obsolete["intent_id"],
                old_target_model_id=target,
                target_model_id=replacement,
            )
        desired = (
            self.switch_intents[0]["target_model_id"]
            if self.switch_intents else automatic_target
        )
        if desired == self.gpu_model:
            desired = None
        if desired != self.proposed_target:
            old = self.proposed_target
            self.proposed_target = desired
            if old is None and desired is not None:
                self._emit("SWITCH_PROPOSED", target_model_id=desired)
            elif old is not None:
                self._emit(
                    "SWITCH_COALESCED",
                    old_target_model_id=old,
                    target_model_id=desired,
                )

    def reconsider(self) -> list[dict[str, Any]]:
        before = len(self.actions)
        while True:
            selected: tuple[Demand, str] | None = None
            for demand in self._queued_demands():
                executor_id = self._available_executor(demand.model_id)
                if executor_id is not None:
                    selected = (demand, executor_id)
                    break
            if selected is None:
                break
            demand, executor_id = selected
            queue = self.queues[demand.model_id]
            require(queue and queue[0] == demand.request_id,
                    "model FIFO order violated")
            queue.popleft()
            executor = self.executors[executor_id]
            executor["credit"] -= 1
            record = self.requests[demand.request_id]
            record["state"] = "IN_FLIGHT"
            record["executor_id"] = executor_id
            self._emit(
                "DISPATCH",
                executor_id=executor_id,
                model_id=demand.model_id,
                request_id=demand.request_id,
            )

        nonterminal = sorted(
            (
                record["demand"]
                for record in self.requests.values()
                if record["state"] in {"QUEUED", "IN_FLIGHT"}
            ),
            key=lambda item: item.arrival_order,
        )
        desired = None
        for demand in nonterminal:
            if demand.model_id == self.gpu_model:
                continue
            desired = demand.model_id
            break

        self._refresh_switch_proposal(desired)
        return self.actions[before:]

    def submit_switch_intent(
            self,
            *,
            intent_id: str,
            target_model_id: str,
            trigger_request_id: str,
            intent_order: int) -> list[dict[str, Any]]:
        require_string(intent_id, "intent_id")
        require(target_model_id in self.models, "switch: unknown target")
        require(
            trigger_request_id in self.requests,
            "switch: unknown trigger request",
        )
        require(
            self.requests[trigger_request_id]["demand"].model_id
            == target_model_id,
            "switch: trigger model mismatch",
        )
        require_int(intent_order, "intent_order")
        require(intent_order > self.last_switch_order,
                "switch: intent order must increase")
        self.last_switch_order = intent_order
        if not self.promotion_enabled:
            self._emit(
                "SWITCH_DISABLED",
                intent_id=intent_id,
                target_model_id=target_model_id,
            )
            return []
        self.switch_intents.append({
            "intent_id": intent_id,
            "intent_order": intent_order,
            "target_model_id": target_model_id,
            "trigger_request_id": trigger_request_id,
        })
        return self.reconsider()

    def start_switch(self) -> dict[str, Any]:
        require(self.transition_target is None, "switch: already active")
        require(self.proposed_target is not None, "switch: no proposal")
        self.transition_target = self.proposed_target
        self.proposed_target = None
        if self.switch_intents \
                and self.switch_intents[0]["target_model_id"] \
                == self.transition_target:
            self.switch_intents.popleft()
        gpu = self.executors["GPU"]
        return self._emit(
            "SWITCH_STARTED",
            from_model_id=self.gpu_model,
            target_model_id=self.transition_target,
        )

    def can_start_switch(self) -> bool:
        return (
            self.proposed_target is not None
            and self.transition_target is None
        )

    def drain_complete(self) -> bool:
        gpu = self.executors["GPU"]
        return (
            self.transition_target is not None
            and gpu["credit"] == gpu["capacity"]
        )

    def finish_switch(self) -> list[dict[str, Any]]:
        require(self.transition_target is not None, "switch: no active target")
        require(self.drain_complete(), "switch: GPU drain incomplete")
        old_model = self.gpu_model
        self.gpu_model = self.transition_target
        self.transition_target = None
        gpu = self.executors["GPU"]
        gpu["model_id"] = self.gpu_model
        gpu["ready"] = True
        gpu["credit"] = gpu["capacity"]
        self._emit(
            "SWITCH_FINISHED",
            from_model_id=old_model,
            target_model_id=self.gpu_model,
        )
        return self.reconsider()

    def complete(self, request_id: str) -> list[dict[str, Any]]:
        record = self.requests.get(request_id)
        require(record is not None, "completion: unknown request")
        require(record["state"] == "IN_FLIGHT", "completion: invalid state")
        executor_id = record["executor_id"]
        executor = self.executors[executor_id]
        require(executor["credit"] < executor["capacity"],
                "completion: executor credit overflow")
        executor["credit"] += 1
        record["state"] = "COMPLETED"
        self._emit(
            "REQUEST_COMPLETED",
            executor_id=executor_id,
            model_id=record["demand"].model_id,
            request_id=request_id,
        )
        return self.reconsider()

    def cancel_queued(self, request_id: str) -> list[dict[str, Any]]:
        record = self.requests.get(request_id)
        require(record is not None and record["state"] == "QUEUED",
                "cancel: request is not queued")
        queue = self.queues[record["demand"].model_id]
        queue.remove(request_id)
        record["state"] = "STRANDED"
        self._emit("REQUEST_STRANDED", request_id=request_id)
        return self.reconsider()

    def finalize(self) -> list[dict[str, Any]]:
        for demand in self._queued_demands():
            record = self.requests[demand.request_id]
            record["state"] = "STRANDED"
            self.queues[demand.model_id].remove(demand.request_id)
            self._emit(
                "REQUEST_STRANDED",
                model_id=demand.model_id,
                request_id=demand.request_id,
            )
        for request_id, record in sorted(self.requests.items()):
            if record["state"] == "IN_FLIGHT":
                executor = self.executors[record["executor_id"]]
                require(
                    executor["credit"] < executor["capacity"],
                    "finalize: executor credit overflow",
                )
                executor["credit"] += 1
                record["state"] = "STRANDED"
                self._emit(
                    "REQUEST_STRANDED",
                    model_id=record["demand"].model_id,
                    request_id=request_id,
                )
        require(
            all(record["credit"] == record["capacity"]
                for record in self.executors.values()),
            "finalize: executor credits not restored",
        )
        return self.actions

    def terminal_counts(self) -> dict[str, int]:
        result = {"COMPLETED": 0, "STRANDED": 0}
        for record in self.requests.values():
            state = record["state"]
            if state in result:
                result[state] += 1
        return result
