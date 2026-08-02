#!/usr/bin/env python3
"""Fail-closed result contract for the S37 physical cut sweep."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from arbitrary_topology import route_specs


SCHEMA = "s37-arbitrary-resident-cut-sweep-v1"


class ContractError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def with_digest(value: dict[str, Any]) -> dict[str, Any]:
    if "result_hash" in value:
        raise ContractError("result already has a digest")
    result = dict(value)
    result["result_hash"] = "sha256:" + hashlib.sha256(
        canonical_bytes(value)
    ).hexdigest()
    return result


def validate(value: object) -> None:
    if type(value) is not dict or value.get("schema") != SCHEMA:
        raise ContractError("schema mismatch")
    supplied = value.get("result_hash")
    if type(supplied) is not str:
        raise ContractError("result digest is missing")
    unhashed = dict(value)
    del unhashed["result_hash"]
    expected = "sha256:" + hashlib.sha256(canonical_bytes(unhashed)).hexdigest()
    if supplied != expected:
        raise ContractError("result digest mismatch")
    if value.get("physical") is not True or value.get("status") != "PASS":
        raise ContractError("physical PASS status is required")
    signature = value.get("token_signature")
    if (
        type(signature) is not list
        or len(signature) != 4
        or any(type(token) is not int for token in signature)
    ):
        raise ContractError("token signature is invalid")

    expected_routes = {
        route_id: (phone, cut) for route_id, phone, cut in route_specs()
    }
    rows = value.get("routes")
    if type(rows) is not list or len(rows) != len(expected_routes):
        raise ContractError("route result set is incomplete")
    seen = set()
    for row in rows:
        if type(row) is not dict:
            raise ContractError("route result must be an object")
        route_id = row.get("route_id")
        if route_id in seen or route_id not in expected_routes:
            raise ContractError("route id is invalid")
        seen.add(route_id)
        phone, cut = expected_routes[route_id]
        if row.get("phone") != phone or row.get("cut") != cut:
            raise ContractError("route identity mismatch")
        repetitions = row.get("repetitions")
        if type(repetitions) is not list or len(repetitions) != 2:
            raise ContractError("two repetitions are required")
        for repetition in repetitions:
            if (
                type(repetition) is not dict
                or repetition.get("batch_size") != 8
                or repetition.get("tokens") != signature
                or type(repetition.get("request_ids")) is not list
                or len(repetition["request_ids"]) != 8
            ):
                raise ContractError("repetition result is invalid")
            events = repetition.get("events")
            if type(events) is not dict:
                raise ContractError("physical events are missing")
            head = events.get(phone)
            tail = events.get("tail")
            if type(head) is not list or not head or type(tail) is not list or not tail:
                raise ContractError("head or tail did not execute")
            if any(
                type(event) is not dict or event.get("active_range") != [0, cut]
                for event in head
            ):
                raise ContractError("phone active range differs from cut")
            if any(
                type(event) is not dict or event.get("active_range") != [cut, 48]
                for event in tail
            ):
                raise ContractError("tail active range differs from cut")
    if seen != set(expected_routes):
        raise ContractError("route result set differs from contract")

    final_state = value.get("final_state")
    if type(final_state) is not dict:
        raise ContractError("final state is missing")
    leases = final_state.get("software_leases")
    active = final_state.get("active_sequences")
    if (
        final_state.get("route_pins") != {}
        or type(leases) is not dict
        or type(active) is not dict
        or any(leases.values())
        or any(active.values())
    ):
        raise ContractError("runtime state was not drained")
