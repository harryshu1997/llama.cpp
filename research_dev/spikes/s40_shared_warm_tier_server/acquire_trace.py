#!/usr/bin/env python3
"""Drive the frozen S40 trace through the asynchronous warm-tier API."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Protocol
from urllib.parse import quote, urlsplit

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
from validate_inputs import DEFAULT_CONTRACT, validate_contract
from event_evidence import validate_event_shape


ADMISSION_KEYS = {
    "arrival_order",
    "controller_epoch",
    "request_id",
    "schema",
    "state",
}
STATUS_KEYS = {
    "committed_output_tokens",
    "controller_epoch",
    "model",
    "owner_id",
    "ownership_epoch",
    "position",
    "publication_index",
    "request_id",
    "schema",
    "state",
}
STATES = {"QUEUED", "ACTIVE", "COMPLETED", "STRANDED"}
TERMINAL_STATES = {"COMPLETED", "STRANDED"}
FINALIZE_KEYS = {
    "controller_epoch",
    "reason",
    "schema",
    "state",
}
FINALIZE_REASONS = {"HORIZON_REACHED", "TRACE_COMPLETE"}
FINALIZE_STATES = {"DRAINING", "FINALIZED"}
FINALIZE_STATUS_KEYS = {
    "controller_epoch",
    "schema",
    "state",
}
TRACE_START_KEYS = {
    "active_drain_deadline_ns",
    "campaign_horizon_ns",
    "campaign_horizon_us",
    "created_ns",
    "drain_bound_us",
    "event_log_run_start_ns",
    "experiment_contract_sha256",
    "host_boot_id",
    "requests_sha256",
    "run_id",
    "runtime_config_sha256",
    "schema",
    "trace_origin_ns",
    "trace_start_lead_us",
}


class TraceTransport(Protocol):
    def admit(self, request: dict[str, Any]) -> dict[str, Any]:
        ...

    def status(self, request_id: str) -> dict[str, Any]:
        ...

    def finalize(self, reason: str) -> dict[str, Any]:
        ...

    def finalization_status(self) -> dict[str, Any]:
        ...


class HttpRecorder:
    def __init__(
            self,
            path: Path,
            run_id: str,
            trace_start_sha256: str) -> None:
        require(path.is_absolute(), "http recorder: path must be absolute")
        require(not path.exists(), "http recorder: output already exists")
        self.path = path
        self.sink = path.open("xb", buffering=0)
        self.run_id = run_id
        self.trace_start_sha256 = validate_digest(
            trace_start_sha256, "http recorder.trace_start_sha256")
        self.sequence = 0
        self.lock = threading.Lock()

    def close(self) -> None:
        self.sink.flush()
        os.fsync(self.sink.fileno())
        self.sink.close()
        directory_fd = os.open(
            self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def append(
            self,
            method: str,
            path: str,
            operation: str,
            request_id: str,
            attempt: int,
            start_ns: int,
            end_ns: int,
            request_raw: bytes,
            status_code: int,
            response_headers: list[tuple[str, str]],
            response_raw: bytes) -> None:
        with self.lock:
            row = {
                "attempt": attempt,
                "http_status": status_code,
                "method": method,
                "operation": operation,
                "path": path,
                "request_body_base64": base64.b64encode(
                    request_raw).decode("ascii"),
                "request_body_sha256": digest_bytes(request_raw),
                "request_id": request_id,
                "response_body_base64": base64.b64encode(
                    response_raw).decode("ascii"),
                "response_body_sha256": digest_bytes(response_raw),
                "response_headers": [
                    [name.lower(), value]
                    for name, value in response_headers
                ],
                "run_id": self.run_id,
                "schema": "s40-http-evidence-v1",
                "sequence": self.sequence,
                "t_end_ns": end_ns,
                "t_start_ns": start_ns,
                "trace_start_sha256": self.trace_start_sha256,
            }
            self.sink.write(canonical_bytes(row))
            self.sequence += 1


class HttpTransport:
    def __init__(
            self,
            base_url: str,
            recorder: HttpRecorder,
            timeout_s: int = 30) -> None:
        parsed = urlsplit(base_url)
        require(
            parsed.scheme == "http"
            and parsed.hostname is not None
            and parsed.path in {"", "/"}
            and not parsed.query
            and not parsed.fragment,
            "HTTP transport: base URL must be an HTTP origin",
        )
        self.host = parsed.hostname
        self.port = parsed.port or 80
        self.recorder = recorder
        self.timeout_s = timeout_s
        self.poll_attempts: dict[str, int] = {}

    def _request(
            self,
            method: str,
            path: str,
            operation: str,
            request_id: str,
            body: dict[str, Any] | None,
            attempt: int) -> dict[str, Any]:
        request_raw = canonical_bytes(body) if body is not None else b""
        connection = http.client.HTTPConnection(
            self.host, self.port, timeout=self.timeout_s)
        start_ns = time.monotonic_ns()
        try:
            connection.request(
                method,
                path,
                body=request_raw if body is not None else None,
                headers=(
                    {"Content-Type": "application/json"}
                    if body is not None else {}
                ),
            )
            response = connection.getresponse()
            response_raw = response.read()
            status_code = response.status
            headers = response.getheaders()
        except OSError as error:
            raise EvidenceError(
                f"HTTP {operation} failed for {request_id}: {error}") from error
        finally:
            end_ns = time.monotonic_ns()
            connection.close()
        self.recorder.append(
            method,
            path,
            operation,
            request_id,
            attempt,
            start_ns,
            end_ns,
            request_raw,
            status_code,
            headers,
            response_raw,
        )
        require(status_code == 200,
                f"HTTP {operation}: status {status_code}")
        value = parse_json(response_raw, f"HTTP {operation} response")
        require(isinstance(value, dict),
                f"HTTP {operation}: expected object response")
        return value

    def admit(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "POST",
            "/experimental/warm-tier/requests",
            "ADMIT",
            request["request_id"],
            request,
            0,
        )

    def status(self, request_id: str) -> dict[str, Any]:
        attempt = self.poll_attempts.get(request_id, 0)
        self.poll_attempts[request_id] = attempt + 1
        return self._request(
            "GET",
            "/experimental/warm-tier/requests/" + quote(
                request_id, safe=""),
            "STATUS",
            request_id,
            None,
            attempt,
        )

    def finalize(self, reason: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/experimental/warm-tier/finalize",
            "FINALIZE",
            "",
            {
                "reason": reason,
                "schema": "llama-server-warm-tier-finalize-v1",
            },
            0,
        )

    def finalization_status(self) -> dict[str, Any]:
        return self._request(
            "GET",
            "/experimental/warm-tier/finalize",
            "FINALIZE_STATUS",
            "",
            None,
            0,
        )


def validate_admission(
        value: Any,
        request_id: str,
        arrival_order: int) -> dict[str, Any]:
    require(isinstance(value, dict), "admission: expected object")
    require(set(value) == ADMISSION_KEYS, "admission: key set mismatch")
    require(
        value["schema"] == "llama-server-warm-tier-admission-v1",
        "admission: schema mismatch",
    )
    require(value["request_id"] == request_id,
            "admission: request ID mismatch")
    require(value["arrival_order"] == arrival_order,
            "admission: arrival order mismatch")
    require_int(value["controller_epoch"], "admission.controller_epoch")
    require(value["state"] in STATES, "admission: invalid state")
    return value


def validate_status(
        value: Any,
        expected: dict[str, Any],
        previous: dict[str, Any] | None) -> dict[str, Any]:
    require(isinstance(value, dict), "status: expected object")
    require(set(value) == STATUS_KEYS, "status: key set mismatch")
    require(
        value["schema"] == "llama-server-warm-tier-request-status-v1",
        "status: schema mismatch",
    )
    require(value["request_id"] == expected["event_id"],
            "status: request ID mismatch")
    require(value["model"] == expected["model_id"],
            "status: model mismatch")
    require(value["state"] in STATES, "status: invalid state")
    tokens = value["committed_output_tokens"]
    require(
        isinstance(tokens, list)
        and all(
            isinstance(token, int)
            and not isinstance(token, bool)
            and 0 <= token < (1 << 31)
            for token in tokens
        )
        and len(tokens) <= expected["output_tokens"],
        "status: invalid committed token history",
    )
    publication = require_int(
        value["publication_index"], "status.publication_index")
    require(publication == len(tokens), "status: publication count mismatch")
    require(
        require_int(value["position"], "status.position")
        == len(expected["prompt_tokens"]) + len(tokens),
        "status: position mismatch",
    )
    require_int(value["controller_epoch"], "status.controller_epoch")
    ownership_epoch = require_int(
        value["ownership_epoch"], "status.ownership_epoch")
    owner = value["owner_id"]
    require(owner is None or (
        isinstance(owner, str) and bool(owner) and owner.isascii()),
        "status: invalid owner")
    if value["state"] in {"ACTIVE", "COMPLETED"}:
        require(owner is not None and ownership_epoch > 0,
                "status: active request lacks owner")
    if value["state"] == "COMPLETED":
        require(
            len(tokens) == expected["output_tokens"],
            "status: completed request has wrong token count",
        )
    if previous is not None:
        old_tokens = previous["committed_output_tokens"]
        require(tokens[:len(old_tokens)] == old_tokens,
                "status: committed history changed")
        require(publication >= previous["publication_index"],
                "status: publication regressed")
        require(ownership_epoch >= previous["ownership_epoch"],
                "status: ownership epoch regressed")
        require(
            value["controller_epoch"] >= previous["controller_epoch"],
            "status: controller epoch regressed",
        )
        require(previous["state"] not in TERMINAL_STATES,
                "status: polled after terminal")
    return value


def validate_finalize(value: Any, reason: str) -> dict[str, Any]:
    require(reason in FINALIZE_REASONS, "finalize: invalid reason")
    require(isinstance(value, dict), "finalize: expected object")
    require(set(value) == FINALIZE_KEYS, "finalize: key set mismatch")
    require(
        value["schema"] == "llama-server-warm-tier-finalize-result-v1",
        "finalize: schema mismatch",
    )
    require(value["reason"] == reason, "finalize: reason mismatch")
    require(value["state"] in FINALIZE_STATES, "finalize: invalid state")
    require_int(value["controller_epoch"], "finalize.controller_epoch")
    return value


def validate_finalize_status(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "finalize status: expected object")
    require(
        set(value) == FINALIZE_STATUS_KEYS,
        "finalize status: key set mismatch",
    )
    require(
        value["schema"] == "llama-server-warm-tier-finalize-status-v1",
        "finalize status: schema mismatch",
    )
    require(
        value["state"] in {"OPEN", "DRAINING", "FINALIZED", "FAILED"},
        "finalize status: invalid state",
    )
    require_int(
        value["controller_epoch"], "finalize status.controller_epoch")
    return value


def wait_until(
        target_ns: int,
        clock: Callable[[], int],
        sleep: Callable[[float], None]) -> int:
    while True:
        now = clock()
        if now >= target_ns:
            return now
        sleep((target_ns - now) / 1_000_000_000)


def run_trace(
        requests: list[dict[str, Any]],
        origin_ns: int,
        transport: TraceTransport,
        *,
        campaign_horizon_ns: int,
        drain_bound_ns: int,
        clock: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval_ns: int = 20_000_000,
        poll_workers: int = 32) -> dict[str, Any]:
    require(bool(requests), "trace: empty request list")
    require(poll_interval_ns >= 1_000_000,
            "trace: poll interval below 1 ms")
    require(1 <= poll_workers <= 128, "trace: invalid poll worker count")
    require(campaign_horizon_ns > origin_ns,
            "trace: invalid campaign horizon")
    require(drain_bound_ns > 0, "trace: invalid active-drain bound")
    first_target = origin_ns + requests[0]["arrival_us"] * 1000
    require(clock() <= first_target, "trace: first admission deadline missed")

    admission_lateness_ns = []
    for arrival_order, expected in enumerate(requests):
        target_ns = origin_ns + expected["arrival_us"] * 1000
        actual_ns = wait_until(target_ns, clock, sleep)
        request = {
            "arrival_order": arrival_order,
            "max_output_tokens": 8,
            "model": expected["model_id"],
            "prompt_tokens": expected["prompt_tokens"],
            "request_id": expected["event_id"],
            "schema": "llama-server-warm-tier-request-v1",
        }
        admission = validate_admission(
            transport.admit(request),
            expected["event_id"],
            arrival_order,
        )
        require(
            admission["state"] not in {"COMPLETED", "STRANDED"},
            "admission: request terminal before acknowledgement",
        )
        admission_lateness_ns.append(actual_ns - target_ns)

    expected_by_id = {row["event_id"]: row for row in requests}
    statuses: dict[str, dict[str, Any] | None] = {
        request_id: None for request_id in expected_by_id
    }
    terminal: dict[str, dict[str, Any]] = {}
    require(
        campaign_horizon_ns
        > origin_ns + max(row["arrival_us"] for row in requests) * 1000,
        "trace: campaign horizon precedes final arrival",
    )
    finalization_reason: str | None = None
    finalization_state: str | None = None
    drain_deadline_ns: int | None = None
    while len(terminal) != len(requests) \
            or finalization_reason is not None \
            and finalization_state != "FINALIZED":
        now = clock()
        if finalization_reason is None and now >= campaign_horizon_ns:
            finalization_reason = "HORIZON_REACHED"
            finalization = validate_finalize(
                transport.finalize(finalization_reason),
                finalization_reason,
            )
            finalization_state = finalization["state"]
            drain_deadline_ns = campaign_horizon_ns + drain_bound_ns
        if drain_deadline_ns is not None:
            require(
                now <= drain_deadline_ns,
                "trace: active-drain bound exceeded",
            )
        pending = [
            request_id for request_id in expected_by_id
            if request_id not in terminal
        ]
        if pending:
            with ThreadPoolExecutor(max_workers=poll_workers) as pool:
                futures = {
                    request_id: pool.submit(transport.status, request_id)
                    for request_id in pending
                }
                for request_id in pending:
                    status = validate_status(
                        futures[request_id].result(),
                        expected_by_id[request_id],
                        statuses[request_id],
                    )
                    statuses[request_id] = status
                    if status["state"] in TERMINAL_STATES:
                        terminal[request_id] = status
        if finalization_reason is not None \
                and finalization_state != "FINALIZED":
            finalization = validate_finalize_status(
                transport.finalization_status())
            require(
                finalization["state"] not in {"OPEN", "FAILED"},
                "trace: finalization did not remain fail-closed",
            )
            finalization_state = finalization["state"]
            if finalization_state == "FINALIZED":
                require(
                    drain_deadline_ns is None
                    or clock() <= drain_deadline_ns,
                    "trace: finalization completed after active-drain bound",
                )
        if len(terminal) != len(requests) \
                or finalization_reason is not None \
                and finalization_state != "FINALIZED":
            sleep(poll_interval_ns / 1_000_000_000)

    if finalization_reason is None:
        finalization_reason = "TRACE_COMPLETE"
        finalization = validate_finalize(
            transport.finalize(finalization_reason),
            finalization_reason,
        )
        finalization_state = finalization["state"]
        require(finalization_state == "FINALIZED",
                "trace: completed run did not finalize synchronously")
    else:
        require(finalization_state == "FINALIZED",
                "trace: finalization did not reach FINALIZED")
    return {
        "admission_count": len(requests),
        "completed_count": sum(
            status["state"] == "COMPLETED" for status in terminal.values()),
        "campaign_horizon_ns": campaign_horizon_ns,
        "drain_bound_ns": drain_bound_ns,
        "finalization_reason": finalization_reason,
        "finalization_state": finalization_state,
        "maximum_admission_lateness_ns": max(admission_lateness_ns),
        "schema": "s40-trace-acquisition-result-v2",
        "stranded_count": sum(
            status["state"] == "STRANDED" for status in terminal.values()),
        "terminal_count": len(terminal),
    }


def load_run_start_event(
        event_log: Path,
        expected_run_id: str) -> dict[str, Any]:
    require(event_log.is_file(), "trace: event log is missing")
    with event_log.open("rb") as source:
        first_raw = source.readline()
    require(first_raw.endswith(b"\n"), "trace: incomplete RUN_START row")
    first = parse_json(first_raw, "trace.run_start")
    require(
        isinstance(first, dict),
        "trace: invalid RUN_START",
    )
    validate_event_shape(first, 0, expected_run_id, -1, 0)
    require(
        first["kind"] == "run_start",
        "trace: invalid RUN_START",
    )
    return first


def load_run_start(
        event_log: Path,
        runtime_config: Path,
        expected_run_id: str) -> dict[str, Any]:
    first = load_run_start_event(event_log, expected_run_id)
    config_digest = digest_file(runtime_config)
    require(
        first.get("runtime_config_sha256") == config_digest,
        "trace: RUN_START runtime config digest mismatch",
    )
    validate_digest(config_digest, "trace.runtime_config_sha256")
    require(first.get("run_id") == expected_run_id,
            "trace: RUN_START run ID mismatch")
    return {
        "run_id": expected_run_id,
        "runtime_config_sha256": config_digest,
        "t_ns": require_int(first.get("t_ns"), "trace.run_start.t_ns"),
    }


def write_trace_start(
        path: Path,
        *,
        run_start: dict[str, Any],
        contract_path: Path,
        requests_sha256: str,
        campaign_horizon_us: int,
        drain_bound_us: int,
        trace_start_lead_us: int,
        host_boot_id: str,
        clock: Callable[[], int] = time.monotonic_ns) -> dict[str, Any]:
    require(path.is_absolute(), "trace start: path must be absolute")
    require(not path.exists(), "trace start: output already exists")
    require(
        isinstance(host_boot_id, str)
        and host_boot_id
        and host_boot_id.isascii(),
        "trace start: invalid host boot ID",
    )
    created_ns = clock()
    origin_ns = created_ns + trace_start_lead_us * 1000
    value = {
        "active_drain_deadline_ns": (
            origin_ns + (campaign_horizon_us + drain_bound_us) * 1000
        ),
        "campaign_horizon_ns": origin_ns + campaign_horizon_us * 1000,
        "campaign_horizon_us": campaign_horizon_us,
        "created_ns": created_ns,
        "drain_bound_us": drain_bound_us,
        "event_log_run_start_ns": run_start["t_ns"],
        "experiment_contract_sha256": digest_file(contract_path),
        "host_boot_id": host_boot_id,
        "requests_sha256": validate_digest(
            requests_sha256, "trace_start.requests_sha256"),
        "run_id": run_start["run_id"],
        "runtime_config_sha256": run_start["runtime_config_sha256"],
        "schema": "s40-trace-start-v1",
        "trace_origin_ns": origin_ns,
        "trace_start_lead_us": trace_start_lead_us,
    }
    require(set(value) == TRACE_START_KEYS, "trace start: key set mismatch")
    raw = canonical_bytes(value)
    with path.open("xb", buffering=0) as sink:
        sink.write(raw)
        sink.flush()
        os.fsync(sink.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return value


def write_new_durable(path: Path, raw: bytes, field: str) -> None:
    require(path.is_absolute(), f"{field}: path must be absolute")
    require(not path.exists(), f"{field}: output already exists")
    with path.open("xb", buffering=0) as sink:
        sink.write(raw)
        sink.flush()
        os.fsync(sink.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--http-evidence", type=Path, required=True)
    parser.add_argument("--trace-start", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    recorder = None
    try:
        validated = validate_contract(args.contract)
        contract = read_json(args.contract, "contract")
        requests = read_jsonl(validated["requests_path"], "requests")
        run_start = load_run_start(
            args.event_log, args.runtime_config, args.run_id)
        try:
            host_boot_id = Path(
                "/proc/sys/kernel/random/boot_id").read_text(
                    encoding="ascii").strip()
        except OSError as error:
            raise EvidenceError(
                f"trace start: cannot read host boot ID: {error}") from error
        workload = contract["workload"]
        trace_start = write_trace_start(
            args.trace_start,
            run_start=run_start,
            contract_path=args.contract,
            requests_sha256=workload["requests_sha256"],
            campaign_horizon_us=workload["campaign_horizon_us"],
            drain_bound_us=workload["drain_bound_us"],
            trace_start_lead_us=workload["trace_start_lead_us"],
            host_boot_id=host_boot_id,
        )
        recorder = HttpRecorder(
            args.http_evidence,
            args.run_id,
            digest_file(args.trace_start),
        )
        transport = HttpTransport(args.base_url, recorder)
        result = run_trace(
            requests,
            trace_start["trace_origin_ns"],
            transport,
            campaign_horizon_ns=trace_start["campaign_horizon_ns"],
            drain_bound_ns=workload["drain_bound_us"] * 1000,
        )
        result["trace_start_sha256"] = digest_file(args.trace_start)
        write_new_durable(
            args.result, canonical_bytes(result), "result")
    except (EvidenceError, OSError, http.client.HTTPException) as error:
        print(f"ERROR: {error}")
        return 2
    finally:
        if recorder is not None:
            recorder.close()
    print(canonical_bytes(result).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
