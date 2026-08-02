#!/usr/bin/env python3
"""Parse opt-in StageV3 direct-relay activation evidence."""

from __future__ import annotations

import hashlib
import json


PREFIX = b"DIRECTFRAME "
SCHEMA = "ls-stage-direct-frame-v1"
KEYS = {
    "activation_payload_bytes",
    "call_index",
    "hidden_width",
    "payload_sha256",
    "positions",
    "request_ids",
    "route_epochs",
    "rows",
    "schema",
    "seq_ids",
}
SHAPE_KEYS = {
    "hidden_width",
    "positions",
    "request_ids",
    "route_epochs",
    "rows",
    "seq_ids",
}


class DirectFrameError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise DirectFrameError(message)


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise DirectFrameError(f"E_JSON_NUMBER: {value}")


def canonical_bytes(value):
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError) as error:
        raise DirectFrameError("E_CANONICAL") from error


def integer(value, field, minimum=0):
    require(type(value) is int and value >= minimum, f"E_INTEGER: {field}")
    return value


def digest(value, field):
    require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"E_DIGEST: {field}",
    )
    return value


def integer_vector(value, field, rows, minimum):
    require(
        type(value) is list and len(value) == rows,
        f"E_VECTOR: {field}",
    )
    for index, item in enumerate(value):
        integer(item, f"{field}[{index}]", minimum)
    return value


def parse_record(raw, field):
    require(raw.endswith(b"\n"), f"E_CANONICAL: {field}")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DirectFrameError(f"E_JSON: {field}") from error
    require(type(value) is dict and set(value) == KEYS, f"E_KEYS: {field}")
    require(canonical_bytes(value) == raw, f"E_CANONICAL: {field}")
    require(value["schema"] == SCHEMA, f"E_SCHEMA: {field}")
    call_index = integer(value["call_index"], f"{field}.call_index")
    rows = integer(value["rows"], f"{field}.rows", 1)
    require(rows <= 4096, f"E_RANGE: {field}.rows")
    hidden_width = integer(value["hidden_width"], f"{field}.hidden_width", 1)
    require(hidden_width <= 65536, f"E_RANGE: {field}.hidden_width")
    payload_bytes = integer(
        value["activation_payload_bytes"],
        f"{field}.activation_payload_bytes",
        1,
    )
    require(
        payload_bytes == rows * hidden_width * 4,
        f"E_PAYLOAD_BYTES: {field}",
    )
    digest(value["payload_sha256"], f"{field}.payload_sha256")
    integer_vector(value["request_ids"], f"{field}.request_ids", rows, 1)
    integer_vector(value["route_epochs"], f"{field}.route_epochs", rows, 1)
    integer_vector(value["seq_ids"], f"{field}.seq_ids", rows, 0)
    integer_vector(value["positions"], f"{field}.positions", rows, 0)
    lineage = list(
        zip(
            value["request_ids"],
            value["route_epochs"],
            value["seq_ids"],
            value["positions"],
        )
    )
    require(len(lineage) == len(set(lineage)), f"E_LINEAGE_REUSE: {field}")
    seq_owners = {}
    request_owners = {}
    for request_id, route_epoch, seq_id, _ in lineage:
        owner = (request_id, route_epoch)
        require(
            seq_id not in seq_owners or seq_owners[seq_id] == owner,
            f"E_LINEAGE_REUSE: {field}",
        )
        require(
            request_id not in request_owners
            or request_owners[request_id] == (route_epoch, seq_id),
            f"E_LINEAGE_REUSE: {field}",
        )
        seq_owners[seq_id] = owner
        request_owners[request_id] = (route_epoch, seq_id)
    return value, call_index


def parse_direct_frames(raw, expected_calls=None, payloads=None):
    require(type(raw) is bytes and bool(raw), "E_RAW")
    records = []
    for line_index, line in enumerate(raw.splitlines(keepends=True)):
        if line.startswith(PREFIX):
            value, call_index = parse_record(
                line[len(PREFIX):],
                f"direct_frame[{line_index}]",
            )
            require(
                call_index == len(records),
                f"E_CALL_INDEX: direct_frame[{line_index}]",
            )
            records.append(value)
        elif PREFIX.rstrip() in line:
            raise DirectFrameError(f"E_PREFIX: direct_frame[{line_index}]")
    require(bool(records), "E_DIRECT_FRAME_MISSING")

    if expected_calls is not None:
        require(
            type(expected_calls) is list and len(expected_calls) == len(records),
            "E_EXPECTED_CALL_COUNT",
        )
        for index, (record, expected) in enumerate(
                zip(records, expected_calls)):
            require(
                type(expected) is dict and set(expected) == SHAPE_KEYS,
                f"E_EXPECTED_CALL: {index}",
            )
            actual = {key: record[key] for key in SHAPE_KEYS}
            require(actual == expected, f"E_CALL_SHAPE: {index}")

    if payloads is not None:
        require(
            type(payloads) is list and len(payloads) == len(records),
            "E_PAYLOAD_COUNT",
        )
        for index, (record, payload) in enumerate(zip(records, payloads)):
            require(type(payload) is bytes, f"E_PAYLOAD_TYPE: {index}")
            require(
                len(payload) == record["activation_payload_bytes"],
                f"E_PAYLOAD_LENGTH: {index}",
            )
            require(
                hashlib.sha256(payload).hexdigest()
                == record["payload_sha256"],
                f"E_PAYLOAD_SHA256: {index}",
            )
    return records
