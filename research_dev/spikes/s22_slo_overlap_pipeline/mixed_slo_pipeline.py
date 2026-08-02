#!/usr/bin/env python3
"""Execute a finite mixed-SLO route set against live StageNet V3 workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from dataclasses import asdict
from pathlib import Path

from async_pipeline import (
    DeviceBatcher,
    RequestSpec,
    parse_endpoint,
    run_request,
    summarize_batches,
)
from slo_policy import RouteProfile, SloRouter, WorkRequest
from stage_v3_client import ProtocolError, STAGE_V3_CAP_TERMINAL, StageV3Client


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"), object_pairs_hook=_reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def require_numeric_override(
    scopes: dict[str, str], route_ids: list[str], allow: bool,
) -> None:
    selected_scopes = {scopes[route_id] for route_id in route_ids}
    if "NUMERICALLY_UNCERTIFIED" in selected_scopes and not allow:
        raise ValueError(
            "selected route is numerically uncertified; mechanics-only override required"
        )


def load_profiles(path: Path) -> tuple[list[RouteProfile], dict[str, str]]:
    value = _load_json(path)
    if value.get("schema") != "s22-route-profiles-v1":
        raise ValueError("route profile schema mismatch")
    rows = value.get("routes")
    if not isinstance(rows, list) or not rows:
        raise ValueError("route profiles must be a nonempty list")
    profiles: list[RouteProfile] = []
    scopes: dict[str, str] = {}
    expected = {
        "route_id", "head_name", "offloaded_layers", "fixed_us",
        "p95_step_us", "profiled_batch", "max_active", "gather_cap_us",
        "wait_stages_per_step", "evidence_sha256", "correctness_scope",
        "evidence_path",
    }
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError("route profile keys mismatch")
        scope = row["correctness_scope"]
        if scope not in ("EXACT_POINT", "NUMERICALLY_UNCERTIFIED"):
            raise ValueError("route correctness scope is invalid")
        evidence_path = row["evidence_path"]
        if not isinstance(evidence_path, str) or not evidence_path:
            raise ValueError("route evidence path must be nonempty")
        profile_root = path.parent.resolve()
        artifact = (profile_root / evidence_path).resolve()
        if not artifact.is_relative_to(profile_root):
            raise ValueError("route evidence path escapes the profile directory")
        if not artifact.is_file():
            raise ValueError(f"route evidence artifact is missing: {evidence_path}")
        artifact_sha256 = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
        if artifact_sha256 != row["evidence_sha256"]:
            raise ValueError(f"route evidence digest mismatch: {evidence_path}")
        profile = RouteProfile(**{
            key: row[key] for key in expected
            if key not in ("correctness_scope", "evidence_path")
        })
        profiles.append(profile)
        scopes[profile.route_id] = str(scope)
    return profiles, scopes


def load_trace(path: Path) -> list[WorkRequest]:
    value = _load_json(path)
    if value.get("schema") != "s22-mixed-slo-trace-v1":
        raise ValueError("mixed trace schema mismatch")
    rows = value.get("requests")
    if not isinstance(rows, list) or not rows:
        raise ValueError("mixed trace requests must be a nonempty list")
    expected = {"request_id", "arrival_us", "slo_us", "steps", "priority"}
    requests: list[WorkRequest] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError("mixed trace request keys mismatch")
        requests.append(WorkRequest(**row))
    ids = [request.request_id for request in requests]
    if len(ids) != len(set(ids)):
        raise ValueError("mixed trace request ids must be unique")
    return sorted(requests, key=lambda request: (
        request.arrival_us, request.priority, request.request_id,
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=parse_endpoint, required=True)
    parser.add_argument("--op15", type=parse_endpoint, required=True)
    parser.add_argument("--op12", type=parse_endpoint, required=True)
    parser.add_argument("--tail", type=parse_endpoint, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--queue-depth", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--token", type=int, default=2)
    parser.add_argument("--session-end", choices=("stop", "detach"), default="stop")
    parser.add_argument("--allow-numeric-uncertified", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.queue_depth <= 0 or args.timeout <= 0:
        parser.error("queue depth and timeout must be positive")

    profiles, correctness_scopes = load_profiles(args.profiles)
    requests = load_trace(args.trace)
    profile_by_route = {profile.route_id: profile for profile in profiles}
    endpoint_names = {"cuda", "op15", "op12"}
    if {profile.head_name for profile in profiles} != endpoint_names:
        raise ValueError("profiles must define exactly cuda, op15, and op12 heads")

    clients: dict[str, StageV3Client] = {}
    batchers: dict[str, DeviceBatcher] = {}
    try:
        for name in sorted(endpoint_names):
            clients[name] = StageV3Client.connect(*getattr(args, name), args.timeout)
        clients["tail"] = StageV3Client.connect(*args.tail, args.timeout)
        hellos = {name: client.hello() for name, client in clients.items()}
        tail_hello = hellos["tail"]
        if not tail_hello.capabilities & STAGE_V3_CAP_TERMINAL:
            raise ProtocolError("mixed pipeline tail is not terminal")
        for name in endpoint_names:
            hello = hellos[name]
            if hello.layer_start != 0 or hello.capabilities & STAGE_V3_CAP_TERMINAL:
                raise ProtocolError(f"{name} is not a prefix worker")
            if hello.layer_end != tail_hello.layer_start:
                raise ProtocolError(f"{name} join boundary differs from the tail")

        router = SloRouter(profiles)
        decisions = []
        route_seq: dict[str, int] = {name: 0 for name in endpoint_names}
        specs: list[RequestSpec] = []
        for request in requests:
            decision = router.admit(request, request.arrival_us)
            profile = profile_by_route[decision.route_id]
            head_seq = route_seq[profile.head_name]
            route_seq[profile.head_name] += 1
            if head_seq >= hellos[profile.head_name].max_streams:
                raise ProtocolError(f"{profile.head_name} sequence capacity exceeded")
            tail_seq = len(specs)
            if tail_seq >= tail_hello.max_streams:
                raise ProtocolError("tail sequence capacity exceeded")
            specs.append(RequestSpec(
                request.request_id, decision.route_epoch, profile.head_name,
                head_seq, tail_seq, request.steps, request.slo_us / 1000.0,
                decision.batch_wait_us,
            ))
            decisions.append(asdict(decision))
        require_numeric_override(
            correctness_scopes,
            [str(decision["route_id"]) for decision in decisions],
            args.allow_numeric_uncertified,
        )

        gather_by_head = {
            profile.head_name: profile.gather_cap_us for profile in profiles
        }
        tail_gather_us = min(profile.gather_cap_us for profile in profiles)
        for name, client in clients.items():
            hello = hellos[name]
            gather_us = tail_gather_us if name == "tail" else gather_by_head[name]
            batchers[name] = DeviceBatcher(
                name, client,
                min(hello.n_batch, hello.n_ubatch),
                gather_us, args.queue_depth,
            )

        start_ns = time.monotonic_ns()
        outcomes: list[dict | None] = [None] * len(specs)
        errors: list[BaseException] = []
        errors_lock = threading.Lock()

        def target(index: int, spec: RequestSpec, request: WorkRequest) -> None:
            try:
                due_ns = start_ns + request.arrival_us * 1000
                remaining_s = (due_ns - time.monotonic_ns()) / 1e9
                if remaining_s > 0:
                    time.sleep(remaining_s)
                outcome = run_request(
                    spec, batchers[spec.head_name], batchers["tail"], args.token,
                    threading.Barrier(1), args.timeout,
                )
                elapsed_from_arrival_ms = (time.monotonic_ns() - due_ns) / 1e6
                outcome["elapsed_ms"] = elapsed_from_arrival_ms
                outcome["slo_met"] = elapsed_from_arrival_ms <= spec.slo_ms
                outcome["arrival_us"] = request.arrival_us
                outcome["priority"] = request.priority
                outcome["route_id"] = router.pinned(request.request_id).route_id
                outcomes[index] = outcome
                router.complete(request.request_id, spec.route_epoch)
            except BaseException as exc:
                with errors_lock:
                    errors.append(exc)

        threads = [
            threading.Thread(target=target, args=(index, spec, request),
                             name=f"mixed-{spec.request_id}")
            for index, (spec, request) in enumerate(zip(specs, requests))
        ]
        for thread in threads:
            thread.start()
        max_steps = max(request.steps for request in requests)
        for thread in threads:
            thread.join(timeout=args.timeout * max_steps)
        if any(thread.is_alive() for thread in threads):
            raise TimeoutError("mixed request threads did not finish")
        if errors:
            raise RuntimeError("mixed pipeline failed") from errors[0]
        completed = [outcome for outcome in outcomes if outcome is not None]
        if len(completed) != len(requests):
            raise RuntimeError("mixed request conservation failed")

        for batcher in batchers.values():
            batcher.stop(args.timeout)
        events = {name: list(batcher.events) for name, batcher in batchers.items()}
        batchers.clear()
        for spec in specs:
            clients[spec.head_name].remove(spec.head_seq, spec.request_id, spec.route_epoch)
            clients["tail"].remove(spec.tail_seq, spec.request_id, spec.route_epoch)
        statuses = {name: client.drain() for name, client in clients.items()}
        if any(status.active_sequences != 0 for status in statuses.values()):
            raise ProtocolError("mixed pipeline left live sequences")
        if any(router.active_counts().values()):
            raise RuntimeError("router active-count conservation failed")

        token_sets = {tuple(outcome["tokens"]) for outcome in completed}
        all_slo_met = all(outcome["slo_met"] for outcome in completed)
        exact_routes = all(
            correctness_scopes[outcome["route_id"]] == "EXACT_POINT"
            for outcome in completed
        )
        if not all_slo_met:
            verdict = "SLO_FAIL"
        elif len(token_sets) == 1 and exact_routes:
            verdict = "PASS"
        else:
            verdict = "MECHANICS_PASS_NUMERICALLY_UNCERTIFIED"
        report = {
            "schema": "s22-mixed-slo-physical-v1",
            "verdict": verdict,
            "mechanics_pass": all_slo_met,
            "cross_route_tokens_equal": len(token_sets) == 1,
            "correctness_scopes": correctness_scopes,
            "numeric_uncertified_override": args.allow_numeric_uncertified,
            "decisions": decisions,
            "requests": completed,
            "workers": {name: asdict(hello) for name, hello in hellos.items()},
            "batches": {name: summarize_batches(rows) for name, rows in events.items()},
            "batch_events": events,
        }
        for client in clients.values():
            if args.session_end == "stop":
                client.stop()
            else:
                client.detach()
        data = json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data, encoding="ascii")
        print(data, end="")
        return 0 if all_slo_met else 2
    finally:
        for batcher in batchers.values():
            try:
                batcher.stop(args.timeout)
            except BaseException:
                pass
        for client in clients.values():
            client.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ProtocolError, RuntimeError, TimeoutError, ValueError) as exc:
        print(json.dumps({"verdict": "FAIL", "error": str(exc)}, sort_keys=True))
        raise SystemExit(2)
