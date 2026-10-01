#!/usr/bin/env python3
"""Compact, read-only summaries of recorded campaign runs for the joint-planner simulator.

One summary per RESULT.json: the request timeline (arrival, execution start, first token, end, phone
share), desktop model loads with their measured energy, phone session re-provisioning (SESSION_LOADING ->
SESSION_READY), the adaptive controller's measured 4-token windows grouped by model / fraction / active
batch, and the trace totals. The summaries are small enough to keep as test fixtures, so the simulator
calibration can be recomputed without the 40 MB run artifacts.

Usage: joint_planner_runs.py OUT.json LABEL=RUN_DIR [LABEL=RUN_DIR ...]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from typing import Any, Mapping

SUMMARY_SCHEMA = "joint-planner-run-summary-v1"
MODEL_SHORT = {"gemma": "gemma", "qwen3": "qwen", "llama": "llama"}


def short_model(model_id: str) -> str:
    for prefix, name in MODEL_SHORT.items():
        if model_id.startswith(prefix):
            return name
    raise ValueError("unknown model id: " + model_id)


def short_request(request_id: str) -> str:
    tail = request_id.split(":", 1)[1] if ":" in request_id else request_id
    return "llama00" if "llama" in tail else tail


def _s(us: float | int | None) -> float | None:
    return None if us is None else round(us / 1e6, 3)


def _joules(domains: Mapping[str, int], key: str) -> float:
    return round(domains.get(key, 0) / 1e6, 1)


def _requests(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    paid_start_ns = result["paid_start_ns"]
    periods = _token_periods(result)
    rows = []
    for row in sorted(result["request_results"], key=lambda r: r["replay_arrival_us"]):
        receipt = row["terminal_ticket"]["execution_receipt"]
        energy = receipt.get("fleet_energy_uj_by_domain") or {}
        proof = row.get("physical_execution_proof") or {}
        layers = proof.get("phone_calls_by_layer") or []
        per_layer = max((item["calls"] for item in layers), default=0)
        first_ns = row.get("first_token_ns")
        rows.append({
            "id": short_request(row["request_id"]),
            "model": short_model(row["model_id"]),
            "arrival_s": _s(row["replay_arrival_us"]),
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "exec_start_s": _s(receipt["started_us"]),
            "first_token_s": None if first_ns is None else round((first_ns - paid_start_ns) / 1e9, 3),
            "end_s": _s(row["completion"]["actual_end_us"]),
            "selected_fraction_ppm": row["fraction_history"].get("selected_split_fraction_ppm"),
            "phone_layers": len(layers),
            "phone_calls_per_layer": per_layer,
            "exec_cpu_j": _joules(energy, "cpu-package"),
            "exec_gpu_j": _joules(energy, "gpu-board"),
            "decode_period_ms": periods.get(row["request_id"]),
        })
    return rows


def _token_periods(result: Mapping[str, Any]) -> dict[str, float]:
    stamps: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for event in result.get("adaptive_timing_events") or []:
        if event.get("kind") == "DECODE_BOUNDARY_OBSERVED":
            stamps[event["request_id"]].append((event["token_index"], event["token_observed_at_us"]))
    periods = {}
    for request_id, values in stamps.items():
        values.sort()
        if len(values) >= 3:
            span = values[-1][1] - values[0][1]
            periods[request_id] = round(span / (len(values) - 1) / 1e3, 1)
    return periods


def _loads(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows, seen = [], set()
    for row in result["request_results"]:
        for ticket in (row["initial_ticket"], row["terminal_ticket"]):
            for receipt in ticket.get("transition_receipts") or []:
                key = (receipt["transition_id"], receipt["started_us"])
                if key in seen or receipt.get("status") != "COMPLETED":
                    continue
                seen.add(key)
                energy = receipt.get("fleet_energy_uj_by_domain") or {}
                model = receipt["transition_id"].split(":")[1]
                rows.append({
                    "model": short_model(model),
                    "device": receipt.get("device_id"),
                    "request": short_request(receipt["request_id"]),
                    "start_s": _s(receipt["started_us"]),
                    "end_s": _s(receipt["finished_us"]),
                    "cpu_j": _joules(energy, "cpu-package"),
                    "gpu_j": _joules(energy, "gpu-board"),
                })
    return sorted(rows, key=lambda r: r["start_s"])


def _phone_sessions(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    sha_to_model = {
        value["artifact_sha256"]: short_model(model_id)
        for model_id, value in result["model_artifacts"].items()
    }
    loading: dict[str, tuple[float, str]] = {}
    rows = []
    for event in result.get("phone_residency_events") or []:
        session = event.get("session") or {}
        endpoint = session.get("endpoint") or ""
        model = sha_to_model.get(session.get("resident_artifact_sha256"))
        if event["kind"] == "SESSION_LOADING":
            loading[endpoint] = (event["observed_at_us"], model)
        elif event["kind"] == "SESSION_READY" and endpoint in loading:
            started, loading_model = loading.pop(endpoint)
            rows.append({
                "session": endpoint.rsplit("/", 1)[-1],
                "model": model or loading_model,
                "loading_s": _s(started),
                "ready_s": _s(event["observed_at_us"]),
            })
    return rows


def _proposals(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for event in result.get("phone_residency_events") or []:
        if event["kind"] in ("PROPOSED", "PROPOSAL_UPDATED"):
            rows.append({
                "kind": event["kind"],
                "at_s": _s(event["observed_at_us"]),
                "reason": event.get("reason"),
            })
    return rows


def _windows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    models = {row["request_id"]: short_model(row["model_id"]) for row in result["request_results"]}
    batch: dict[str, int] = defaultdict(lambda: 1)
    grouped: dict[tuple[str, int, int], list[tuple[float, float]]] = defaultdict(list)
    for event in sorted(result.get("request_helper_events") or [], key=lambda e: e["event_index"]):
        request_id = event.get("request_id")
        if event["kind"] == "CONTEXT_CHANGED":
            batch[request_id] = event.get("active_batch") or 1
        elif event["kind"] == "LEARNING_WINDOW_RECORDED" and request_id in models:
            fraction = event.get("split_fraction_ppm")
            if fraction is None:
                continue
            grouped[(models[request_id], fraction, batch[request_id])].append(
                (event["latency_per_token_us"] / 1e3, event["fleet_energy_uj"] / 1e6)
            )
    rows = []
    for (model, fraction, active_batch), values in sorted(grouped.items()):
        rows.append({
            "model": model,
            "fraction_ppm": fraction,
            "active_batch": active_batch,
            "windows": len(values),
            "latency_ms_p50": round(statistics.median(v[0] for v in values), 1),
            "window_fleet_j_p50": round(statistics.median(v[1] for v in values), 2),
        })
    return rows


def summarize_result(result: Mapping[str, Any], label: str) -> dict[str, Any]:
    domains = result["trace_energy"]["fleet_energy_uj_by_domain"]
    phone_j = sum(value for key, value in domains.items() if "phone" in key) / 1e6
    policy = (result.get("dispatch_policy") or {}).get("policy") or {}
    return {
        "schema": SUMMARY_SCHEMA,
        "label": label,
        "status": result.get("status"),
        "duration_s": _s(result["duration_us"]),
        "host_cpu_kj": round(domains.get("cpu-package", 0) / 1e9, 3),
        "host_gpu_kj": round(domains.get("gpu-board", 0) / 1e9, 3),
        "phone_assumed_kj": round(phone_j / 1e3, 3),
        "dispatch_policy": {k: v for k, v in policy.items() if k != "schema"},
        "adaptive_decode_overrides": dict(result.get("adaptive_decode_overrides") or {}),
        "dispatch_statistics": dict((result.get("dispatch_policy") or {}).get("statistics") or {}),
        "requests": _requests(result),
        "loads": _loads(result),
        "phone_sessions": _phone_sessions(result),
        "phone_proposals": _proposals(result),
        "windows": _windows(result),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("runs", nargs="+", help="LABEL=RUN_DIR (a directory holding RESULT.json)")
    args = parser.parse_args(argv)
    summaries = []
    for item in args.runs:
        label, _, directory = item.partition("=")
        if not label or not directory:
            parser.error("runs must be LABEL=RUN_DIR")
        result = json.loads((Path(directory) / "RESULT.json").read_text(encoding="utf-8"))
        summaries.append(summarize_result(result, label))
    args.output.write_text(
        json.dumps({"schema": SUMMARY_SCHEMA, "runs": summaries}, indent=1, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
