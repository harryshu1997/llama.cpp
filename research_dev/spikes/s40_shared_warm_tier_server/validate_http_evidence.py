#!/usr/bin/env python3
"""Validate the raw S40 HTTP request and response ledger."""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import Any
from urllib.parse import quote

from acquire_trace import (
    TERMINAL_STATES,
    validate_admission,
    validate_finalize,
    validate_finalize_status,
    validate_status,
)
from evidence_common import (
    EvidenceError,
    canonical_bytes,
    digest_bytes,
    digest_file,
    parse_json,
    read_json,
    read_jsonl,
    require,
    require_int,
    require_string,
    validate_digest,
)


HTTP_KEYS = {
    "attempt",
    "http_status",
    "method",
    "operation",
    "path",
    "request_body_base64",
    "request_body_sha256",
    "request_id",
    "response_body_base64",
    "response_body_sha256",
    "response_headers",
    "run_id",
    "schema",
    "sequence",
    "t_end_ns",
    "t_start_ns",
    "trace_start_sha256",
}
RESULT_KEYS = {
    "admission_count",
    "campaign_horizon_ns",
    "completed_count",
    "drain_bound_ns",
    "finalization_reason",
    "finalization_state",
    "maximum_admission_lateness_ns",
    "schema",
    "stranded_count",
    "terminal_count",
    "trace_start_sha256",
}


def _decode_body(row: dict[str, Any], prefix: str, field: str) -> bytes:
    encoded = row[f"{prefix}_body_base64"]
    require(isinstance(encoded, str) and encoded.isascii(),
            f"{field}: invalid {prefix} base64")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise EvidenceError(
            f"{field}: invalid {prefix} base64") from error
    require(
        digest_bytes(raw) == validate_digest(
            row[f"{prefix}_body_sha256"],
            f"{field}.{prefix}_body_sha256",
        ),
        f"{field}: {prefix} body digest mismatch",
    )
    return raw


def _validate_row(
        row: Any,
        index: int,
        run_id: str,
        trace_start_sha256: str,
        created_ns: int,
        drain_deadline_ns: int) -> tuple[bytes, bytes]:
    field = f"http[{index}]"
    require(isinstance(row, dict) and set(row) == HTTP_KEYS,
            f"{field}: key set mismatch")
    require(
        row["schema"] == "s40-http-evidence-v1"
        and row["sequence"] == index
        and row["run_id"] == run_id
        and row["trace_start_sha256"] == trace_start_sha256,
        f"{field}: identity mismatch",
    )
    started = require_int(row["t_start_ns"], f"{field}.t_start_ns")
    completed = require_int(row["t_end_ns"], f"{field}.t_end_ns")
    require(
        created_ns <= started <= completed <= drain_deadline_ns,
        f"{field}: interval outside trace window",
    )
    require(row["http_status"] == 200, f"{field}: non-200 response")
    require_int(row["attempt"], f"{field}.attempt")
    require_string(row["method"], f"{field}.method")
    require_string(row["operation"], f"{field}.operation")
    require_string(row["path"], f"{field}.path")
    require(isinstance(row["request_id"], str),
            f"{field}.request_id: expected string")
    headers = row["response_headers"]
    require(isinstance(headers, list), f"{field}: invalid response headers")
    for header_index, header in enumerate(headers):
        require(
            isinstance(header, list)
            and len(header) == 2
            and all(
                isinstance(value, str)
                and value.isascii()
                and "\r" not in value
                and "\n" not in value
                for value in header
            )
            and header[0] == header[0].lower(),
            f"{field}: invalid response header {header_index}",
        )
    return (
        _decode_body(row, "request", field),
        _decode_body(row, "response", field),
    )


def validate_http_evidence(
        path: Path,
        trace_start_path: Path,
        result_path: Path,
        requests_path: Path,
        expected_run_id: str) -> dict[str, Any]:
    trace_start = read_json(trace_start_path, "trace_start")
    result = read_json(result_path, "trace_result")
    require(set(result) == RESULT_KEYS, "trace_result: key set mismatch")
    require(
        result["schema"] == "s40-trace-acquisition-result-v2",
        "trace_result: schema mismatch",
    )
    trace_digest = digest_file(trace_start_path)
    require(
        result["trace_start_sha256"] == trace_digest,
        "trace_result: trace-start digest mismatch",
    )
    run_id = require_string(trace_start.get("run_id"), "trace_start.run_id")
    require(run_id == expected_run_id, "trace_start: run ID mismatch")
    created_ns = require_int(trace_start.get("created_ns"),
                             "trace_start.created_ns")
    origin_ns = require_int(trace_start.get("trace_origin_ns"),
                            "trace_start.trace_origin_ns")
    horizon_ns = require_int(trace_start.get("campaign_horizon_ns"),
                             "trace_start.campaign_horizon_ns")
    drain_deadline_ns = require_int(
        trace_start.get("active_drain_deadline_ns"),
        "trace_start.active_drain_deadline_ns",
    )
    require(created_ns <= origin_ns < horizon_ns < drain_deadline_ns,
            "trace_start: invalid time ordering")

    requests = read_jsonl(requests_path, "requests")
    require(bool(requests), "requests: empty")
    expected_by_id = {request["event_id"]: request for request in requests}
    require(len(expected_by_id) == len(requests),
            "requests: duplicate event ID")
    rows = read_jsonl(path, "http")
    require(
        path.read_bytes() == b"".join(canonical_bytes(row) for row in rows),
        "http: not canonical JSONL",
    )
    require(len(rows) >= 2 * len(requests) + 1,
            "http: insufficient records")

    previous_status: dict[str, dict[str, Any] | None] = {
        request_id: None for request_id in expected_by_id
    }
    status_attempt: dict[str, int] = {
        request_id: 0 for request_id in expected_by_id
    }
    terminal: dict[str, dict[str, Any]] = {}
    finalize_response = None
    finalize_statuses = []
    admissions = 0
    for index, row in enumerate(rows):
        request_raw, response_raw = _validate_row(
            row,
            index,
            run_id,
            trace_digest,
            created_ns,
            drain_deadline_ns,
        )
        operation = row["operation"]
        response = parse_json(response_raw, f"http[{index}].response")
        if operation == "ADMIT":
            require(index == admissions and admissions < len(requests),
                    "http: admissions are not the exact prefix")
            expected = requests[admissions]
            expected_body = {
                "arrival_order": admissions,
                "max_output_tokens": 8,
                "model": expected["model_id"],
                "prompt_tokens": expected["prompt_tokens"],
                "request_id": expected["event_id"],
                "schema": "llama-server-warm-tier-request-v1",
            }
            require(
                row["method"] == "POST"
                and row["path"] == "/experimental/warm-tier/requests"
                and row["request_id"] == expected["event_id"]
                and row["attempt"] == 0
                and request_raw == canonical_bytes(expected_body),
                "http: admission request mismatch",
            )
            validate_admission(
                response, expected["event_id"], admissions)
            require(
                row["t_start_ns"]
                >= origin_ns + expected["arrival_us"] * 1000,
                "http: admission precedes scheduled arrival",
            )
            admissions += 1
        elif operation == "STATUS":
            request_id = row["request_id"]
            require(
                admissions == len(requests)
                and request_id in expected_by_id
                and request_id not in terminal
                and row["method"] == "GET"
                and row["path"] == (
                    "/experimental/warm-tier/requests/"
                    + quote(request_id, safe="")
                )
                and request_raw == b""
                and row["attempt"] == status_attempt[request_id],
                "http: status request mismatch",
            )
            status_attempt[request_id] += 1
            status = validate_status(
                response,
                expected_by_id[request_id],
                previous_status[request_id],
            )
            previous_status[request_id] = status
            if status["state"] in TERMINAL_STATES:
                terminal[request_id] = status
        elif operation == "FINALIZE":
            require(
                admissions == len(requests)
                and finalize_response is None
                and row["method"] == "POST"
                and row["path"] == "/experimental/warm-tier/finalize"
                and row["request_id"] == ""
                and row["attempt"] == 0,
                "http: finalize request mismatch",
            )
            body = parse_json(request_raw, "http.finalize.request")
            require(
                isinstance(body, dict)
                and set(body) == {"reason", "schema"}
                and body["schema"]
                == "llama-server-warm-tier-finalize-v1",
                "http: finalize body mismatch",
            )
            finalize_response = validate_finalize(response, body["reason"])
        elif operation == "FINALIZE_STATUS":
            require(
                finalize_response is not None
                and row["method"] == "GET"
                and row["path"] == "/experimental/warm-tier/finalize"
                and row["request_id"] == ""
                and row["attempt"] == 0
                and request_raw == b"",
                "http: finalize-status request mismatch",
            )
            status = validate_finalize_status(response)
            require(status["state"] not in {"OPEN", "FAILED"},
                    "http: finalization failed open")
            if finalize_statuses:
                require(
                    not (
                        finalize_statuses[-1]["state"] == "FINALIZED"
                        and status["state"] != "FINALIZED"
                    ),
                    "http: finalization state regressed",
                )
            finalize_statuses.append(status)
        else:
            raise EvidenceError(f"http[{index}]: unknown operation")

    require(admissions == len(requests),
            "http: admission set is incomplete")
    require(set(terminal) == set(expected_by_id),
            "http: terminal status set is incomplete")
    require(finalize_response is not None,
            "http: missing finalization")
    final_state = (
        finalize_statuses[-1]["state"]
        if finalize_statuses else finalize_response["state"]
    )
    require(final_state == "FINALIZED",
            "http: finalization did not complete")
    if finalize_response["state"] == "DRAINING":
        require(bool(finalize_statuses),
                "http: draining finalization has no status evidence")
    require(
        finalize_response["reason"] == result["finalization_reason"]
        and result["finalization_state"] == "FINALIZED",
        "trace_result: finalization mismatch",
    )
    completed = sum(
        status["state"] == "COMPLETED" for status in terminal.values())
    stranded = sum(
        status["state"] == "STRANDED" for status in terminal.values())
    require(
        result["admission_count"] == admissions
        and result["terminal_count"] == len(terminal)
        and result["completed_count"] == completed
        and result["stranded_count"] == stranded
        and result["campaign_horizon_ns"] == horizon_ns
        and result["drain_bound_ns"]
        == trace_start["drain_bound_us"] * 1000,
        "trace_result: count or time mismatch",
    )
    require_int(
        result["maximum_admission_lateness_ns"],
        "trace_result.maximum_admission_lateness_ns",
    )
    return {
        "admission_count": admissions,
        "completed_count": completed,
        "finalization_reason": finalize_response["reason"],
        "http_record_count": len(rows),
        "run_id": run_id,
        "status": "S40_HTTP_EVIDENCE_VALID",
        "stranded_count": stranded,
    }
