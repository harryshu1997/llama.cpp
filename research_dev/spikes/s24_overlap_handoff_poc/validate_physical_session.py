#!/usr/bin/env python3
"""Validate one S24 runtime report against five StageNet session certificates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "s24-physical-session-validation-v1"
RUN_SCHEMA = "s24-fixed-diamond-physical-v1"
WORKERS = {
    "cuda-prefix": {"range": [0, 8], "backend": "CUDA0"},
    "cuda-mid": {"range": [8, 16], "backend": "CUDA0"},
    "op12-prefix": {"range": [0, 8], "backend": "HTP0"},
    "op15-mid": {"range": [8, 16], "backend": "HTP0"},
    "cuda-tail": {"range": [16, 48], "backend": "CUDA0"},
}
ROUTE_RESOURCES = {
    "R0": ("cuda-prefix", "cuda-mid", "cuda-tail"),
    "R1": ("cuda-prefix", "op15-mid", "cuda-tail"),
    "R2": ("op12-prefix", "op15-mid", "cuda-tail"),
}


class ValidationError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot load runtime report: {exc}") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema") != RUN_SCHEMA
        or value.get("status") != "RUN_COMPLETE"
    ):
        raise ValidationError("runtime report is not complete")
    return value


def parse_session_certificates(path: Path) -> list[dict[str, Any]]:
    certificates = []
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"cannot read worker log {path}: {exc}") from exc
    for line in lines:
        marker = line.find("SESSIONCERT ")
        if marker < 0:
            continue
        payload = line[marker + len("SESSIONCERT "):]
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"malformed SESSIONCERT in {path}") from exc
        if not isinstance(value, dict):
            raise ValidationError(f"non-object SESSIONCERT in {path}")
        certificates.append(value)
    if not certificates:
        raise ValidationError(f"no SESSIONCERT records in {path}")
    return certificates


def physical_event_name(name: str) -> str:
    if name == "op15-mid" or name.startswith("op15-mid-r"):
        return "op15-mid"
    return name


def validate_event_lineage(
    run: dict[str, Any],
) -> tuple[dict[str, int], dict[str, Any]]:
    events_by_physical: dict[str, list[dict[str, Any]]] = {
        name: [] for name in WORKERS
    }
    batch_events = run.get("batch_events")
    if not isinstance(batch_events, dict):
        raise ValidationError("runtime batch events are missing")
    for event_name, events in batch_events.items():
        physical = physical_event_name(event_name)
        if physical not in WORKERS or not isinstance(events, list):
            raise ValidationError(f"unknown batch event stream: {event_name}")
        if any(event.get("worker") != event_name for event in events):
            raise ValidationError(f"{event_name} event changed its queue identity")
        events_by_physical[physical].extend(events)

    if run.get("control") == "C1":
        if "op15-mid" in batch_events or not {
            "op15-mid-r1", "op15-mid-r2",
        }.issubset(batch_events):
            raise ValidationError("C1 OP15 queues are not route-isolated")
    elif any(name.startswith("op15-mid-r") for name in batch_events):
        raise ValidationError("treatment created a route-private OP15 queue")
    if "cuda-tail" not in batch_events:
        raise ValidationError("shared CUDA tail queue is missing")

    expected_rows = {}
    row_sets: dict[str, set[tuple[int, int, int]]] = {
        name: set() for name in WORKERS
    }
    sequences: dict[str, dict[tuple[int, int, int], list[int]]] = {
        name: defaultdict(list) for name in WORKERS
    }
    decisions = run["runtime"]["decisions"]
    route_by_request = {
        (int(row["request_id"]), int(row["route_epoch"])): str(row["route_id"])
        for row in decisions
    }
    if len(route_by_request) != len(decisions):
        raise ValidationError("runtime decisions duplicate a request epoch")
    for worker, events in events_by_physical.items():
        rows = 0
        for event in events:
            batch_size = event.get("batch_size")
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValidationError(f"{worker} has invalid batch size")
            vector_fields = (
                "request_ids", "route_epochs", "seq_ids", "positions",
                "routes", "upstream_workers",
            )
            if any(
                not isinstance(event.get(field), list)
                or len(event[field]) != batch_size
                for field in vector_fields
            ):
                raise ValidationError(f"{worker} batch vector length mismatch")
            if event.get("status") != "OK":
                raise ValidationError(f"{worker} contains a failed physical batch")
            if event.get("contributing_routes") != sorted(set(event["routes"])):
                raise ValidationError(f"{worker} contributing route summary differs")
            if event.get("contributing_upstreams") != sorted(
                set(event["upstream_workers"])
            ):
                raise ValidationError(f"{worker} contributing upstream summary differs")
            for index in range(batch_size):
                request_id = event["request_ids"][index]
                route_epoch = event["route_epochs"][index]
                seq_id = event["seq_ids"][index]
                position = event["positions"][index]
                route_id = event["routes"][index]
                if not all(
                    isinstance(value, int) and value >= 0
                    for value in (request_id, route_epoch, seq_id, position)
                ) or request_id <= 0 or route_epoch <= 0:
                    raise ValidationError(f"{worker} row identity is invalid")
                if route_by_request.get((request_id, route_epoch)) != route_id:
                    raise ValidationError(f"{worker} row changed its pinned route")
                if worker not in ROUTE_RESOURCES.get(route_id, ()):
                    raise ValidationError(f"{worker} row is outside route {route_id}")
                if worker in ("cuda-prefix", "op12-prefix"):
                    expected_upstream = "TOKEN_SOURCE"
                elif worker == "cuda-mid":
                    expected_upstream = "cuda-prefix"
                elif worker == "op15-mid":
                    expected_upstream = (
                        "cuda-prefix" if route_id == "R1" else "op12-prefix"
                    )
                else:
                    expected_upstream = (
                        "cuda-mid" if route_id == "R0" else "op15-mid"
                    )
                if event["upstream_workers"][index] != expected_upstream:
                    raise ValidationError(f"{worker} upstream lineage changed")
                key = (request_id, route_epoch, position)
                if key in row_sets[worker]:
                    raise ValidationError(f"{worker} row lineage is duplicated")
                row_sets[worker].add(key)
                sequences[worker][(request_id, route_epoch, seq_id)].append(position)
            rows += batch_size
        expected_rows[worker] = rows
        for identity, positions in sequences[worker].items():
            ordered = sorted(positions)
            if ordered != list(range(len(ordered))):
                raise ValidationError(
                    f"{worker} sequence positions are not contiguous: {identity}"
                )

    request_records = {
        (int(row["request_id"]), int(row["route_epoch"])): row
        for row in run["runtime"]["requests"]
    }
    if set(request_records) != set(route_by_request):
        raise ValidationError("decision and completion identities differ")
    for identity, route_id in route_by_request.items():
        request = request_records[identity]
        if request["route_id"] != route_id:
            raise ValidationError("completed request route differs from its pin")
        positions = set(range(
            int(request["prompt_length"]) + int(request["output_steps"]) - 1
        ))
        expected = {(identity[0], identity[1], position) for position in positions}
        for worker in ROUTE_RESOURCES[route_id]:
            observed = {
                key for key in row_sets[worker] if key[:2] == identity
            }
            if observed != expected:
                raise ValidationError(
                    f"{worker} lineage differs for request {identity[0]}"
                )
        for worker in set(WORKERS) - set(ROUTE_RESOURCES[route_id]):
            if any(key[:2] == identity for key in row_sets[worker]):
                raise ValidationError(
                    f"request {identity[0]} executed outside pinned route"
                )

    final_software = run.get("final_software_state")
    if not isinstance(final_software, dict):
        raise ValidationError("final software state is missing")
    if final_software.get("runner_pins"):
        raise ValidationError("route pins remain live")
    leases = final_software.get("software_leases")
    if not isinstance(leases, dict) or any(leases.values()):
        raise ValidationError("software leases remain live")
    for worker, status in run.get("final_workers", {}).items():
        if worker not in WORKERS:
            raise ValidationError("unknown final worker status")
        if (
            status.get("before_drain", {}).get("active_sequences") != 0
            or status.get("after_drain", {}).get("active_sequences") != 0
            or status.get("after_drain", {}).get("draining") is not True
        ):
            raise ValidationError(f"{worker} did not drain physical KV")
    return expected_rows, {
        "request_count": len(request_records),
        "position_continuity": "PASS",
        "route_pin_continuity": "PASS",
        "software_leases_zero": "PASS",
        "worker_kv_zero": "PASS",
    }


def validate_backend_tally(
    worker: str,
    tally: dict[str, Any],
    expected_backend: str,
) -> list[dict[str, Any]]:
    if not isinstance(tally, dict) or not tally:
        raise ValidationError(f"{worker} has no compute placement tally")
    exceptions = []
    for operation, backends in tally.items():
        if not isinstance(operation, str) or not isinstance(backends, dict) or not backends:
            raise ValidationError(f"{worker} placement tally is malformed")
        for backend, count in backends.items():
            if not isinstance(count, int) or count <= 0:
                raise ValidationError(f"{worker} placement count is invalid")
            if backend == expected_backend:
                continue
            allowed = operation == "GET_ROWS" and (
                (expected_backend == "HTP0" and backend == "CPU")
                or (expected_backend == "CUDA0" and backend == "CUDA_Host")
            )
            if not allowed:
                raise ValidationError(
                    f"{worker} undeclared compute placement: {operation}@{backend}"
                )
            exceptions.append({
                "operation": operation,
                "backend": backend,
                "count": count,
                "classification": "DECLARED_METADATA_GET_ROWS",
            })
    return exceptions


def validate_certificate(
    worker: str,
    certificate: dict[str, Any],
    expected_rows: int,
    session_end: str,
) -> dict[str, Any]:
    expected = WORKERS[worker]
    if (
        certificate.get("schema") != "ls-stagenet-session-v2"
        or certificate.get("proto_version") != 2
        or certificate.get("session_end") != session_end.upper()
        or certificate.get("expected_backend") != expected["backend"]
        or [certificate.get("layer_start"), certificate.get("layer_end")]
        != expected["range"]
        or certificate.get("n_layer") != 48
        or certificate.get("steps_session") != expected_rows
        or certificate.get("missing_buffer_compute_nodes") != 0
        or certificate.get("reset_applied") is not (session_end == "detach")
        or not isinstance(certificate.get("worker_pid"), int)
        or not certificate.get("worker_boot_nonce")
        or not certificate.get("device_boot_id")
    ):
        raise ValidationError(f"{worker} session certificate contract mismatch")
    if expected_rows == 0:
        if (
            certificate.get("placement_status") != "PLACEMENT_UNOBSERVED"
            or certificate.get("compute_by_op_and_buffer") != {}
        ):
            raise ValidationError(f"idle {worker} reported unexpected compute")
        exceptions = []
    else:
        if certificate.get("placement_status") != "SCHEDULED_PLACEMENT_OK":
            raise ValidationError(f"{worker} placement did not pass")
        exceptions = validate_backend_tally(
            worker,
            certificate.get("compute_by_op_and_buffer"),
            expected["backend"],
        )
    return {
        "expected_rows": expected_rows,
        "worker_pid": certificate["worker_pid"],
        "worker_boot_nonce": certificate["worker_boot_nonce"],
        "device_boot_id": certificate["device_boot_id"],
        "steps_total": certificate["steps_total"],
        "placement_status": certificate["placement_status"],
        "declared_metadata_exceptions": exceptions,
    }


def validate_session(
    runtime_path: Path,
    session_ids: int | dict[str, int],
    log_paths: dict[str, Path],
) -> dict[str, Any]:
    if isinstance(session_ids, int):
        session_ids = {worker: session_ids for worker in WORKERS}
    if (
        set(session_ids) != set(WORKERS)
        or any(session_id <= 0 for session_id in session_ids.values())
        or set(log_paths) != set(WORKERS)
    ):
        raise ValidationError("session identity or log set is invalid")
    run = load_run(runtime_path)
    expected_rows, lineage = validate_event_lineage(run)
    session_end = run["configuration"]["session_end"]
    worker_results = {}
    log_records = {}
    for worker, path in log_paths.items():
        session_id = session_ids[worker]
        certificates = parse_session_certificates(path)
        matches = [
            certificate for certificate in certificates
            if certificate.get("session_id") == session_id
        ]
        if len(matches) != 1:
            raise ValidationError(
                f"{worker} session {session_id} certificate count is {len(matches)}"
            )
        worker_results[worker] = validate_certificate(
            worker,
            matches[0],
            expected_rows[worker],
            session_end,
        )
        log_records[worker] = {
            "path": str(path),
            "sha256": "sha256:" + sha256_file(path),
        }
    return {
        "schema": SCHEMA,
        "status": "PHYSICAL_SESSION_PASS",
        "runtime": {
            "path": str(runtime_path),
            "sha256": "sha256:" + sha256_file(runtime_path),
            "control": run["control"],
        },
        "session_ids": session_ids,
        "session_end": session_end,
        "expected_rows": expected_rows,
        "lineage": lineage,
        "workers": worker_results,
        "logs": log_records,
        "placement_gate": "PASS",
        "cpu_compute_fallback": "NONE_EXCEPT_DECLARED_METADATA_GET_ROWS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--session-id", type=int)
    parser.add_argument("--cuda-session-id", type=int)
    parser.add_argument("--phone-session-id", type=int)
    parser.add_argument("--cuda-prefix-log", type=Path, required=True)
    parser.add_argument("--cuda-mid-log", type=Path, required=True)
    parser.add_argument("--op12-log", type=Path, required=True)
    parser.add_argument("--op15-log", type=Path, required=True)
    parser.add_argument("--cuda-tail-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    logs = {
        "cuda-prefix": args.cuda_prefix_log,
        "cuda-mid": args.cuda_mid_log,
        "op12-prefix": args.op12_log,
        "op15-mid": args.op15_log,
        "cuda-tail": args.cuda_tail_log,
    }
    cuda_session_id = args.cuda_session_id or args.session_id
    phone_session_id = args.phone_session_id or args.session_id
    if cuda_session_id is None or phone_session_id is None:
        parser.error(
            "pass --session-id or both --cuda-session-id and --phone-session-id"
        )
    session_ids = {
        "cuda-prefix": cuda_session_id,
        "cuda-mid": cuda_session_id,
        "op12-prefix": phone_session_id,
        "op15-mid": phone_session_id,
        "cuda-tail": cuda_session_id,
    }
    try:
        report = validate_session(args.runtime, session_ids, logs)
    except (OSError, ValidationError, ValueError) as exc:
        print(json.dumps({
            "status": "FAIL", "error": str(exc),
        }, sort_keys=True, separators=(",", ":")))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(report))
    print(json.dumps({
        "status": report["status"],
        "session_ids": report["session_ids"],
        "output": str(args.output),
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
