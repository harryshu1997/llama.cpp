#!/usr/bin/env python3
"""Derive the S26 scheduler profile from frozen S24 physical runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from priority_policy import PriorityRoute, RouteBatchPoint


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CP4 = REPO / "research_dev/spikes/s24_overlap_handoff_poc/results/cp4_fixed_diamond"
MANIFEST = CP4 / "SHA256SUMS.txt"
MANIFEST_SHA256 = "54437990ee47247903d89f6ba5c404281982903c03eb0adfc207e2c53e85ac5c"
SCHEMA = "s26-priority-route-points-v2"
TRACE_HASH = "sha256:9b4e84bdd5d6bf38bd8d951dd043ad5d7a1b22520f746ec2953ab547db81c8be"
OUTPUT_TOKENS = [532, 236772, 236772, 564]
OUTPUT_TOKENS_BY_POINT = {
    ("R0", 1): OUTPUT_TOKENS,
    ("R0", 4): [532, 236772, 236772, 107],
    ("R2", 1): OUTPUT_TOKENS,
    ("R2", 4): OUTPUT_TOKENS,
}

SOURCE_SPECS = (
    ("r0-b1-a", "R0", 1, "e4ea3f1dc77b243056a1556b4a4dab43c66568348cc7e8b04dbcbef66551f8ea"),
    ("r0-b1-b", "R0", 1, "7cd8d6f54f512e8cbf7c7b4db2f56fdd564d57bd81464103d418c371fa43e78f"),
    ("r2-b1-a", "R2", 1, "022bb14aba2522ce0b54af5fb5cb81914c83ca09a2d24c79ad1909d124d15c14"),
    ("r2-b1-b", "R2", 1, "47c525a12e0e602f9db0b0802a93bc57e5282bf35b263934b37a66b2972411a5"),
)

LOCKSTEP_SPECS = (
    {
        "id": "r0-b4-lockstep",
        "route_id": "R0",
        "control": "C0",
        "runtime": HERE / "results/r0b4_route_point/runtime.json",
        "runtime_sha256": "d78c3ca404bc5475327c5dc34d474c96e51ccbb9350bdd50446a276b25021880",
        "validation": HERE / "results/r0b4_route_point/session-validation-portable.json",
        "validation_sha256": "497cc1dbc9bbf307c274630b4eaa293099890ad46fd2fc556c9071ca1552972d",
    },
    {
        "id": "r2-b4-lockstep",
        "route_id": "R2",
        "control": "C2",
        "runtime": HERE / "results/r2b4_route_point/runtime.json",
        "runtime_sha256": "1f75c23bdd908898c946d56d0d8f422662c84d6b0a4aa3f62172d62c2ac5d538",
        "validation": HERE / "results/r2b4_route_point/session-validation-portable.json",
        "validation_sha256": "364c7c60b1fbc572ea935cc02a80c92ea2f7cf314adbd6c430f1cd1732b0df99",
    },
)

WORKER_SHAPES = {
    "cuda-prefix": (0, 8, 4),
    "cuda-mid": (8, 16, 4),
    "op12-prefix": (0, 8, 4),
    "op15-mid": (8, 16, 4),
    "cuda-tail": (16, 48, 8),
}

ACTIVE_WORKERS = {
    "R0": {"cuda-prefix", "cuda-mid", "cuda-tail"},
    "R2": {"op12-prefix", "op15-mid", "cuda-tail"},
}


class PriorityProfileError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PriorityProfileError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"), object_pairs_hook=_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PriorityProfileError(f"cannot read {path}: {exc}") from exc
    if type(value) is not dict:
        raise PriorityProfileError(f"{path} must contain a JSON object")
    return value


def _manifest_entries() -> dict[str, str]:
    if sha256_file(MANIFEST) != MANIFEST_SHA256:
        raise PriorityProfileError("frozen S24 manifest digest changed")
    result = {}
    try:
        lines = MANIFEST.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PriorityProfileError(f"cannot read S24 manifest: {exc}") from exc
    for line in lines:
        digest, separator, name = line.partition("  ")
        if not separator or len(digest) != 64 or name in result:
            raise PriorityProfileError("invalid S24 manifest line")
        result[name.removeprefix("./")] = digest
    return result


def _source_path(name: str) -> Path:
    return CP4 / "runs" / name / "runtime.json"


def _validate_source(
    report: dict[str, Any], route_id: str, batch_size: int, source_id: str,
    expected_control: str = "C2",
) -> dict[str, int]:
    if report.get("schema") != "s24-fixed-diamond-physical-v1":
        raise PriorityProfileError(f"{source_id}: runtime schema changed")
    if (
        report.get("status") != "RUN_COMPLETE"
        or report.get("control") != expected_control
    ):
        raise PriorityProfileError(f"{source_id}: source run control changed")
    if report.get("trace", {}).get("trace_hash") != TRACE_HASH:
        raise PriorityProfileError(f"{source_id}: trace identity changed")
    expected_scope = (
        "Q8_CUDA_ONLY"
        if route_id == "R0"
        else "MECHANICS_ONLY_F16_PHONE_Q8_SERVER_NUMERICALLY_UNCERTIFIED"
    )
    if report.get("numeric_scope") != expected_scope:
        raise PriorityProfileError(f"{source_id}: numeric scope changed")

    summary = report.get("summary")
    runtime = report.get("runtime")
    if type(summary) is not dict or type(runtime) is not dict:
        raise PriorityProfileError(f"{source_id}: summary or runtime is absent")
    if (
        summary.get("completed_requests") != batch_size
        or summary.get("rejected_requests") != 0
        or summary.get("slo_misses") != 0
        or summary.get("route_distribution") != {route_id: batch_size}
        or runtime.get("selected_request_count") != batch_size
        or runtime.get("completed_count") != batch_size
        or runtime.get("rejected_count") != 0
    ):
        raise PriorityProfileError(f"{source_id}: request accounting changed")

    requests = runtime.get("requests")
    if type(requests) is not list or len(requests) != batch_size:
        raise PriorityProfileError(f"{source_id}: request records changed")
    for request in requests:
        if (
            type(request) is not dict
            or request.get("route_id") != route_id
            or request.get("prompt_length") != 1
            or request.get("output_steps") != 4
            or request.get("output_tokens") != OUTPUT_TOKENS_BY_POINT[(route_id, batch_size)]
            or request.get("slo_met") is not True
            or type(request.get("latency_us")) is not int
            or request["latency_us"] <= 0
        ):
            raise PriorityProfileError(f"{source_id}: request shape or result changed")

    workers = report.get("workers")
    if type(workers) is not dict or set(workers) != set(WORKER_SHAPES):
        raise PriorityProfileError(f"{source_id}: worker set changed")
    for worker, (start, end, streams) in WORKER_SHAPES.items():
        hello = workers[worker]
        if (
            type(hello) is not dict
            or hello.get("layer_start") != start
            or hello.get("layer_end") != end
            or hello.get("max_streams") != streams
        ):
            raise PriorityProfileError(f"{source_id}: {worker} shape changed")

    worker_summary = summary.get("workers")
    if type(worker_summary) is not dict or set(worker_summary) != set(WORKER_SHAPES):
        raise PriorityProfileError(f"{source_id}: worker summary changed")
    for worker, row in worker_summary.items():
        expected_sizes = [batch_size] * 4 if worker in ACTIVE_WORKERS[route_id] else []
        if (
            type(row) is not dict
            or row.get("batch_sizes") != expected_sizes
            or row.get("failed_batches") != 0
        ):
            raise PriorityProfileError(f"{source_id}: {worker} batch evidence changed")

    final_workers = report.get("final_workers")
    if type(final_workers) is not dict or set(final_workers) != set(WORKER_SHAPES):
        raise PriorityProfileError(f"{source_id}: final worker set changed")
    for worker, row in final_workers.items():
        if row.get("after_drain", {}).get("active_sequences") != 0:
            raise PriorityProfileError(f"{source_id}: {worker} did not drain")
    state = report.get("final_software_state")
    if (
        type(state) is not dict
        or state.get("runner_pins") != {}
        or any(state.get("software_leases", {}).get(name) != {} for name in WORKER_SHAPES)
    ):
        raise PriorityProfileError(f"{source_id}: software state did not drain")

    makespan = summary.get("makespan_us")
    cuda_work = summary.get("summed_cuda_island_compute_us")
    if (
        type(makespan) is not int
        or makespan <= 0
        or type(cuda_work) is not int
        or cuda_work <= 0
    ):
        raise PriorityProfileError(f"{source_id}: timing values are invalid")
    return {
        "max_latency_us": max(request["latency_us"] for request in requests),
        "makespan_us": makespan,
        "cuda_work_us": cuda_work,
    }


def _validate_lockstep_evidence(
    spec: dict[str, Any], runtime_sha256: str,
) -> str:
    path = spec["validation"]
    if sha256_file(path) != spec["validation_sha256"]:
        raise PriorityProfileError(f"{spec['id']}: validation artifact changed")
    report = _load_json(path)
    route_id = spec["route_id"]
    active = ACTIVE_WORKERS[route_id]
    expected_rows = {
        worker: 16 if worker in active else 0 for worker in WORKER_SHAPES
    }
    if (
        report.get("schema") != "s24-physical-session-validation-v1"
        or report.get("status") != "PHYSICAL_SESSION_PASS"
        or report.get("placement_gate") != "PASS"
        or report.get("session_end") != "stop"
        or report.get("session_ids") != {worker: 1 for worker in WORKER_SHAPES}
        or report.get("expected_rows") != expected_rows
        or report.get("runtime", {}).get("sha256") != "sha256:" + runtime_sha256
    ):
        raise PriorityProfileError(f"{spec['id']}: physical validation changed")
    workers = report.get("workers")
    if type(workers) is not dict or set(workers) != set(WORKER_SHAPES):
        raise PriorityProfileError(f"{spec['id']}: validated worker set changed")
    for worker, expected in expected_rows.items():
        expected_status = "SCHEDULED_PLACEMENT_OK" if expected else "PLACEMENT_UNOBSERVED"
        if (
            workers[worker].get("expected_rows") != expected
            or workers[worker].get("placement_status") != expected_status
        ):
            raise PriorityProfileError(f"{spec['id']}: worker validation changed")
    logs = report.get("logs")
    if type(logs) is not dict or set(logs) != set(WORKER_SHAPES):
        raise PriorityProfileError(f"{spec['id']}: physical logs are missing")
    for worker, record in logs.items():
        log_path = Path(record.get("path", ""))
        if not log_path.is_absolute():
            log_path = REPO / log_path
        expected = record.get("sha256")
        if expected != "sha256:" + sha256_file(log_path):
            raise PriorityProfileError(f"{spec['id']}: {worker} log changed")
    return spec["validation_sha256"]


def derive_bundle() -> dict[str, Any]:
    manifest = _manifest_entries()
    sources = []
    values = {}
    for source_id, route_id, batch_size, expected_digest in SOURCE_SPECS:
        relative = f"runs/{source_id}/runtime.json"
        if manifest.get(relative) != expected_digest:
            raise PriorityProfileError(f"{source_id}: frozen manifest entry changed")
        path = _source_path(source_id)
        observed = sha256_file(path)
        if observed != expected_digest:
            raise PriorityProfileError(f"{source_id}: physical artifact digest changed")
        report = _load_json(path)
        values[source_id] = _validate_source(report, route_id, batch_size, source_id)
        sources.append({
            "id": source_id,
            "path": str(path.relative_to(REPO)),
            "sha256": "sha256:" + observed,
        })

    lockstep_validation = {}
    for spec in LOCKSTEP_SPECS:
        path = spec["runtime"]
        observed = sha256_file(path)
        if observed != spec["runtime_sha256"]:
            raise PriorityProfileError(f"{spec['id']}: physical artifact changed")
        report = _load_json(path)
        values[spec["id"]] = _validate_source(
            report, spec["route_id"], 4, spec["id"], spec["control"],
        )
        lockstep_validation[spec["id"]] = _validate_lockstep_evidence(
            spec, observed,
        )
        sources.extend((
            {
                "id": spec["id"],
                "path": str(path.relative_to(REPO)),
                "sha256": "sha256:" + observed,
            },
            {
                "id": spec["id"] + "-placement",
                "path": str(spec["validation"].relative_to(REPO)),
                "sha256": "sha256:" + lockstep_validation[spec["id"]],
            },
        ))

    r0_ids = ("r0-b1-a", "r0-b1-b")
    r2_b1_ids = ("r2-b1-a", "r2-b1-b")
    r0_duration = max(values[name]["max_latency_us"] for name in r0_ids)
    r0_cuda = min(values[name]["cuda_work_us"] for name in r0_ids)
    r2_b1_duration = max(values[name]["max_latency_us"] for name in r2_b1_ids)
    r2_b1_cuda = max(values[name]["cuda_work_us"] for name in r2_b1_ids)
    r0_b4 = values["r0-b4-lockstep"]
    r0_b4_duration = max(r0_b4["max_latency_us"], r0_b4["makespan_us"])
    r0_b4_cuda = r0_b4["cuda_work_us"]
    r2_b4 = values["r2-b4-lockstep"]
    r2_b4_duration = max(
        r2_b4["max_latency_us"], r2_b4["makespan_us"],
    )
    r2_b4_cuda = r2_b4["cuda_work_us"]

    digest_by_id = {row["id"]: row["sha256"] for row in sources}
    return {
        "schema": SCHEMA,
        "scope": "MECHANICS_ONLY_F16_PHONE_Q8_SERVER_NUMERICALLY_UNCERTIFIED",
        "work": {"input_tokens": 1, "output_steps": 4, "synthetic_token": 2},
        "manifest": {
            "path": str(MANIFEST.relative_to(REPO)),
            "sha256": "sha256:" + MANIFEST_SHA256,
        },
        "sources": sources,
        "capacities": {
            "cuda-prefix": 4,
            "cuda-mid": 4,
            "op12-prefix": 4,
            "op15-mid": 4,
            "cuda-tail": 8,
        },
        "urgent_reserve": {
            "cuda-prefix": 4,
            "cuda-mid": 4,
            "op12-prefix": 0,
            "op15-mid": 0,
            "cuda-tail": 4,
        },
        "routes": [
            {
                "route_id": "R0",
                "resources": ["cuda-prefix", "cuda-mid", "cuda-tail"],
                "gather_us": 5000,
                "points": [
                    {
                        "batch_size": 1,
                        "input_tokens": 1,
                        "output_steps": 4,
                        "duration_us": r0_duration,
                        "cuda_work_us": r0_cuda,
                        "evidence_sha256": [digest_by_id[name] for name in r0_ids],
                        "derivation": "max repeated B1 latency; min repeated CUDA work baseline",
                    },
                    {
                        "batch_size": 4,
                        "input_tokens": 1,
                        "output_steps": 4,
                        "duration_us": r0_b4_duration,
                        "cuda_work_us": r0_b4_cuda,
                        "evidence_sha256": [
                            digest_by_id["r0-b4-lockstep"],
                            digest_by_id["r0-b4-lockstep-placement"],
                        ],
                        "derivation": "isolated lockstep B4 run with physical placement validation",
                    },
                ],
            },
            {
                "route_id": "R2",
                "resources": ["op12-prefix", "op15-mid", "cuda-tail"],
                "gather_us": 5000,
                "points": [
                    {
                        "batch_size": 1,
                        "input_tokens": 1,
                        "output_steps": 4,
                        "duration_us": r2_b1_duration,
                        "cuda_work_us": r2_b1_cuda,
                        "evidence_sha256": [digest_by_id[name] for name in r2_b1_ids],
                        "derivation": "max repeated B1 latency and CUDA work",
                    },
                    {
                        "batch_size": 4,
                        "input_tokens": 1,
                        "output_steps": 4,
                        "duration_us": r2_b4_duration,
                        "cuda_work_us": r2_b4_cuda,
                        "evidence_sha256": [
                            digest_by_id["r2-b4-lockstep"],
                            digest_by_id["r2-b4-lockstep-placement"],
                        ],
                        "derivation": "isolated lockstep B4 run with physical placement validation",
                    },
                ],
            },
        ],
    }


def load_bundle(
    path: Path,
) -> tuple[tuple[PriorityRoute, ...], dict[str, int], dict[str, int], dict[str, Any]]:
    expected = derive_bundle()
    actual = _load_json(path)
    if actual != expected or path.read_bytes() != canonical_bytes(expected):
        raise PriorityProfileError("priority profile bundle is not the canonical derived bundle")
    routes = tuple(
        PriorityRoute(
            route_id=row["route_id"],
            resources=tuple(row["resources"]),
            points=tuple(RouteBatchPoint(
                batch_size=point["batch_size"],
                input_tokens=point["input_tokens"],
                output_steps=point["output_steps"],
                duration_us=point["duration_us"],
                cuda_work_us=point["cuda_work_us"],
                evidence_sha256=tuple(point["evidence_sha256"]),
                derivation=point["derivation"],
            ) for point in row["points"]),
            gather_us=row["gather_us"],
        )
        for row in actual["routes"]
    )
    return routes, dict(actual["capacities"]), dict(actual["urgent_reserve"]), actual
