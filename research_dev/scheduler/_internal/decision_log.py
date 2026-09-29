"""Canonical append-only records for runtime scheduling decisions."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import threading
from typing import Mapping

from .types import canonical_json


DECISION_LOG_SCHEMA = "research-scheduler-decision-log-v1"
DECISION_EVENT_KINDS = frozenset({
    "ACQUIRED",
    "CANCELLED",
    "COMPLETED",
    "DECISION",
    "FAILED",
    "FALLBACK",
    "REPLAN",
})
TERMINAL_EVENT_KINDS = frozenset({"CANCELLED", "COMPLETED", "FAILED"})


class DecisionLogError(ValueError):
    pass


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class _DecisionLogCheckpoint:
    record_count: int
    terminal_request_ids: frozenset[str]
    attempt_ticket_ids: frozenset[str]
    acquired_ticket_ids: frozenset[str]


class RuntimeDecisionLog:
    """Validate and hash runtime records before making them visible."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: list[bytes] = []
        self._record_hashes: list[str] = []
        self._terminal_request_ids: set[str] = set()
        self._attempt_ticket_ids: set[str] = set()
        self._acquired_ticket_ids: set[str] = set()

    def checkpoint(self) -> object:
        with self._lock:
            return _DecisionLogCheckpoint(
                record_count=len(self._records),
                terminal_request_ids=frozenset(self._terminal_request_ids),
                attempt_ticket_ids=frozenset(self._attempt_ticket_ids),
                acquired_ticket_ids=frozenset(self._acquired_ticket_ids),
            )

    def restore(self, checkpoint: object) -> None:
        if (
            not isinstance(checkpoint, _DecisionLogCheckpoint)
            or checkpoint.record_count < 0
        ):
            raise DecisionLogError("decision log checkpoint is invalid")
        with self._lock:
            if checkpoint.record_count > len(self._records):
                raise DecisionLogError("decision log checkpoint is invalid")
            del self._records[checkpoint.record_count:]
            del self._record_hashes[checkpoint.record_count:]
            self._terminal_request_ids = set(
                checkpoint.terminal_request_ids
            )
            self._attempt_ticket_ids = set(checkpoint.attempt_ticket_ids)
            self._acquired_ticket_ids = set(checkpoint.acquired_ticket_ids)

    @staticmethod
    def _validate_body(body: Mapping[str, object]) -> None:
        required = {
            "attempt_index",
            "causal_input_sha256",
            "candidates",
            "decision_kind",
            "decision_reason",
            "event_kind",
            "event_time_us",
            "lifecycle_state",
            "model_sha256",
            "previous_ticket_id",
            "profile_sha256",
            "request_ids",
            "request_sha256",
            "runtime_snapshot_sha256",
            "selected",
            "ticket_id",
        }
        if set(body) != required:
            raise DecisionLogError("decision log record fields differ")
        event_kind = body["event_kind"]
        if event_kind not in DECISION_EVENT_KINDS:
            raise DecisionLogError("decision log event kind is invalid")
        if type(body["event_time_us"]) is not int or body["event_time_us"] < 0:
            raise DecisionLogError("decision log event time is invalid")
        if type(body["attempt_index"]) is not int or body["attempt_index"] < 0:
            raise DecisionLogError("decision log attempt index is invalid")
        request_ids = body["request_ids"]
        if (
            type(request_ids) is not list
            or not request_ids
            or request_ids != sorted(request_ids)
            or len(request_ids) != len(set(request_ids))
            or any(
                type(request_id) is not str
                or not request_id
                or not request_id.isascii()
                for request_id in request_ids
            )
        ):
            raise DecisionLogError("decision log request ids are invalid")
        for name in ("decision_kind", "lifecycle_state", "ticket_id"):
            value = body[name]
            if type(value) is not str or not value or not value.isascii():
                raise DecisionLogError(f"decision log {name} is invalid")

    def append(
        self,
        body: Mapping[str, object],
        *,
        return_record: bool = True,
        precanonical_body: bool = False,
    ) -> dict[str, object] | None:
        if not isinstance(body, Mapping):
            raise DecisionLogError("decision log body must be a mapping")
        if type(return_record) is not bool:
            raise DecisionLogError(
                "decision log return-record flag is invalid"
            )
        if type(precanonical_body) is not bool:
            raise DecisionLogError(
                "decision log precanonical flag is invalid"
            )
        row = dict(body)
        self._validate_body(row)
        with self._lock:
            request_ids = tuple(row["request_ids"])
            if any(
                request_id in self._terminal_request_ids
                for request_id in request_ids
            ):
                raise DecisionLogError(
                    "decision log cannot append after a terminal record"
                )
            event_kind = row["event_kind"]
            ticket_id = row["ticket_id"]
            if event_kind in {"DECISION", "REPLAN", "FALLBACK"}:
                if ticket_id in self._attempt_ticket_ids:
                    raise DecisionLogError(
                        "decision log attempt is duplicated"
                    )
            elif event_kind == "ACQUIRED":
                if (
                    ticket_id not in self._attempt_ticket_ids
                    or ticket_id in self._acquired_ticket_ids
                ):
                    raise DecisionLogError(
                        "decision log acquisition is invalid"
                    )
            record = {
                "schema": DECISION_LOG_SCHEMA,
                "sequence_index": len(self._records),
                "previous_record_sha256": (
                    "0" * 64
                    if not self._records
                    else self._record_hashes[-1]
                ),
                **row,
            }
            encode = (
                lambda value: json.dumps(
                    value,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("ascii")
                if precanonical_body
                else canonical_json(value).encode("ascii")
            )
            unsigned = encode(record)
            record_sha256 = hashlib.sha256(unsigned).hexdigest()
            if event_kind in {"DECISION", "REPLAN", "FALLBACK"}:
                self._attempt_ticket_ids.add(ticket_id)
            elif event_kind == "ACQUIRED":
                self._acquired_ticket_ids.add(ticket_id)
            elif event_kind in TERMINAL_EVENT_KINDS:
                self._terminal_request_ids.update(request_ids)
            marker = b',"runtime_snapshot_sha256":'
            marker_index = unsigned.find(marker)
            if marker_index < 0:
                raise DecisionLogError(
                    "decision log canonical marker is absent"
                )
            encoded = (
                unsigned[:marker_index]
                + b',"record_sha256":"'
                + record_sha256.encode("ascii")
                + b'"'
                + unsigned[marker_index:]
            )
            self._records.append(encoded)
            self._record_hashes.append(record_sha256)
            return json.loads(encoded) if return_record else None

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "head_record_sha256": (
                    "0" * 64
                    if not self._records
                    else self._record_hashes[-1]
                ),
                "records": [
                    json.loads(record.decode("ascii"))
                    for record in self._records
                ],
                "schema": DECISION_LOG_SCHEMA,
            }

    def canonical_bytes(self) -> bytes:
        with self._lock:
            head = (
                "0" * 64
                if not self._record_hashes
                else self._record_hashes[-1]
            )
            return (
                b'{"head_record_sha256":"'
                + head.encode("ascii")
                + b'","records":['
                + b",".join(self._records)
                + b'],"schema":"'
                + DECISION_LOG_SCHEMA.encode("ascii")
                + b'"}\n'
            )

    def has_acquired(self, ticket_id: str) -> bool:
        if type(ticket_id) is not str or not ticket_id:
            raise DecisionLogError("decision log ticket id is invalid")
        with self._lock:
            return ticket_id in self._acquired_ticket_ids

    @classmethod
    def validate(cls, value: object) -> None:
        if type(value) is not dict or set(value) != {
            "head_record_sha256", "records", "schema"
        }:
            raise DecisionLogError("decision log document fields differ")
        if value["schema"] != DECISION_LOG_SCHEMA:
            raise DecisionLogError("decision log schema differs")
        rows = value["records"]
        if type(rows) is not list:
            raise DecisionLogError("decision log records must be a list")
        previous = "0" * 64
        terminals: set[str] = set()
        attempts: set[str] = set()
        acquired: set[str] = set()
        for index, raw in enumerate(rows):
            if type(raw) is not dict:
                raise DecisionLogError("decision log record must be an object")
            record = copy.deepcopy(raw)
            supplied = record.pop("record_sha256", None)
            if (
                record.pop("schema", None) != DECISION_LOG_SCHEMA
                or record.pop("sequence_index", None) != index
                or record.pop("previous_record_sha256", None) != previous
            ):
                raise DecisionLogError("decision log chain metadata differs")
            cls._validate_body(record)
            expected_record = {
                "schema": DECISION_LOG_SCHEMA,
                "sequence_index": index,
                "previous_record_sha256": previous,
                **record,
            }
            if supplied != _digest(expected_record):
                raise DecisionLogError("decision log record hash differs")
            request_ids = tuple(record["request_ids"])
            if terminals.intersection(request_ids):
                raise DecisionLogError(
                    "decision log contains a post-terminal record"
                )
            event_kind = record["event_kind"]
            ticket_id = record["ticket_id"]
            if event_kind in {"DECISION", "REPLAN", "FALLBACK"}:
                if ticket_id in attempts:
                    raise DecisionLogError("decision log attempt is duplicated")
                attempts.add(ticket_id)
            elif event_kind == "ACQUIRED":
                if ticket_id not in attempts or ticket_id in acquired:
                    raise DecisionLogError(
                        "decision log acquisition is invalid"
                    )
                acquired.add(ticket_id)
            elif event_kind in TERMINAL_EVENT_KINDS:
                terminals.update(request_ids)
            previous = supplied
        expected_head = "0" * 64 if not rows else previous
        if value["head_record_sha256"] != expected_head:
            raise DecisionLogError("decision log head differs")
