#!/usr/bin/env python3
"""WS20: do two phones slow each other down when they serve the same layer at the same time?

    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.parallel_split_microbench plan    CONFIG.json
    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.parallel_split_microbench smoke   CONFIG.json OUT
    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.parallel_split_microbench run     CONFIG.json OUT [--only a,b]
    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.parallel_split_microbench analyze OUT --meter METER.json \\
        [--json CONCURRENCY.json] [--md CONCURRENCY.md]

The planner's parallel column split (``layer_placement_opgroups``: two helpers compute disjoint column shares of
one layer's FFN at once; the desktop waits for the slower share) is priced from each phone's SOLO call time. This
harness measures what concurrency itself costs: the two deployed workers (the OP15's HTP worker over ``adb
forward``, the Pixel's packed worker over the AOA bridge, each launched through the scheduler's own session
class) are driven at the serving cadence in four arms, every arm at rows 1/2/4:

  solo-full  one phone at a time, full-width calls (the WS16 microbenchmark's phone legs; the planner's inputs)
  solo-half  one phone at a time, half-width calls (``columns`` = n_ff/2: what one share of a 50/50 split costs)
  pair-half  both phones at once, half width, every call pair released from one barrier (a 50/50 split)
  pair-full  both phones at once, full width (pure link/host interference, independent of the share)

Per call: host RPC time and the worker's compute time (EXECUTE response). ``analyze`` reports, per phone and row
count, solo vs concurrent RPC and compute, the ratio (``degradation``: concurrent compute / solo compute at the same
width), the pair time (max of the two concurrent RPCs) against max of the two solo half RPCs (``penalty_ms``), and
each phone's energy per call above its idle (``tools/rig/meter_phones.sh --charging controlled`` around ``run``,
the WS16 convention). The verdict is PASS when degradation <= ``max_degradation`` and penalty <= ``max_penalty_ms``;
the planner reads CONCURRENCY.json through ``layer_placement opgroups --concurrency-from`` and only then may put a
parallel split into an EXECUTABLE plan (a prior penalty can only propose).

Timing only: both phones compute the first ``columns`` of their own shard (the workers take a width, not an
offset), which costs the same as disjoint shares; outputs are hash-checked, never compared. The OP15 runs over
adb-TCP here, not the production FunctionFS path (its RPC overhead is an upper bound; its compute is the same).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import socket
import statistics
import struct
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from research_dev.scheduler.campaigns.burstgpt.tools import weight_stream_microbench as wsm

SCHEMA = "ws20-parallel-split-microbench-v1"
RESULT_SCHEMA = "ws20-parallel-split-concurrency-v1"
ARMS = ("solo-full", "solo-half", "pair-half", "pair-full")
PAIR_ARMS = ("pair-half", "pair-full")


def need(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit("config: " + message)


# ---------------------------------------------------------------- configuration

def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text())
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    need(config.get("schema") == SCHEMA, "schema must be " + SCHEMA)
    cadence = config.get("cadence", {})
    for key in ("calls_per_token", "gap_call_ms", "gap_token_ms", "warmup_tokens"):
        need(key in cadence and cadence[key] >= 0, "cadence." + key + " is required")
    need(cadence["calls_per_token"] >= 1, "cadence.calls_per_token must be >= 1")
    rows = config.get("rows", [1, 2, 4])
    need(bool(rows) and all(type(r) is int and 1 <= r <= 64 for r in rows), "rows must be small positive integers")
    need(config.get("segment_seconds", 0) > 0 and config.get("idle_s", 0) > 0, "segment_seconds and idle_s > 0")
    arms = config.get("arms", list(ARMS))
    need(bool(arms) and all(arm in ARMS for arm in arms) and len(set(arms)) == len(arms),
         "arms must be distinct names from " + ",".join(ARMS))
    phones = config.get("phones", [])
    need(len(phones) == 2, "exactly two phones (the pair)")
    names = [phone.get("name") for phone in phones]
    need(all(names) and len(set(names)) == 2, "phones need two distinct names")
    for phone in phones:
        need(phone.get("device") in ("op15", "pixel"), phone["name"] + ": device must be op15 or pixel")
        need(phone.get("transport") in ("adb-tcp", "aoa-bridge"), phone["name"] + ": transport adb-tcp|aoa-bridge")
        need(phone.get("protocol_version") in (5, 6), phone["name"] + ": protocol_version must be 5 or 6")
        need(phone.get("activation") in ("swiglu", "geglu"), phone["name"] + ": activation swiglu|geglu")
        for key in ("layer", "n_embd", "n_ff", "est_call_ms"):
            need(key in phone, f"{phone['name']}: {key} is required")
        need(all(str(r) in phone["est_call_ms"] for r in rows), phone["name"] + ": est_call_ms needs every row count")
        need(bool(phone.get("evidence")) or bool(phone.get("worker")), phone["name"] + ": evidence or worker")
    need(len({phone["n_ff"] for phone in phones}) == 1, "both phones must serve the same n_ff")
    quantum = int(config.get("column_quantum", 32))
    half = phones[0]["n_ff"] // 2
    need(phones[0]["n_ff"] % 2 == 0 and half % quantum == 0, "n_ff / 2 must be a multiple of column_quantum")
    for key, default in (("max_degradation", 1.10), ("max_penalty_ms", 1.0)):
        need(float(config.get(key, default)) > 0, key + " must be > 0")


def arm_width(config: Mapping[str, Any], arm: str) -> int:
    n_ff = int(config["phones"][0]["n_ff"])
    return n_ff // 2 if arm.endswith("-half") else n_ff


def leg_for(config: Mapping[str, Any], phone: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """A WS16-shaped phone leg of ``phone`` for ``arm``; pair arms share one token count (the slower estimate)."""
    leg = {"name": f"{arm}:{phone['name']}", "kind": "phone", "op": "ffn", **phone}
    if arm in PAIR_ARMS:
        others = config["phones"]
        leg["est_call_ms"] = {str(r): max(float(p["est_call_ms"][str(r)]) for p in others) for r in config.get("rows", [1, 2, 4])}
    return leg


def plan(config: Mapping[str, Any]) -> dict[str, Any]:
    rows = config.get("rows", [1, 2, 4])
    entries, total = [], 0.0
    for arm in config.get("arms", list(ARMS)):
        legs = [leg_for(config, phone, arm) for phone in config["phones"]]
        if arm in PAIR_ARMS:
            seconds = 2 * config["idle_s"] + len(rows) * (config["segment_seconds"] + config["cadence"]["warmup_tokens"]
                      * wsm.token_cycle_s(config, float(legs[0]["est_call_ms"]["1"]), "ffn")) + 120
            entries.append({"arm": arm, "phones": [leg["name"] for leg in legs], "seconds": round(seconds),
                            "budget_calls": wsm.phone_budget(config, legs[0]),
                            "tokens_per_segment": {str(r): wsm.phone_tokens(config, legs[0], r) for r in rows}})
            total += seconds + config.get("cooldown_s", 10)
        else:
            for leg in legs:
                seconds = 2 * config["idle_s"] + len(rows) * (config["segment_seconds"] + config["cadence"]["warmup_tokens"]
                          * wsm.token_cycle_s(config, float(leg["est_call_ms"]["1"]), "ffn")) + 90
                entries.append({"arm": arm, "phones": [leg["name"]], "seconds": round(seconds),
                                "budget_calls": wsm.phone_budget(config, leg),
                                "tokens_per_segment": {str(r): wsm.phone_tokens(config, leg, r) for r in rows}})
                total += seconds + config.get("cooldown_s", 10)
    return {"schema": SCHEMA, "legs": entries, "total_s": round(total), "total_min": round(total / 60, 1)}


# ---------------------------------------------------------------- driving

def execute_width(stream: socket.socket, leg: Mapping[str, Any], request_id: int, payload: bytes, rows: int,
                  columns: int, clock: Callable[[], int]) -> tuple[int, int, int]:
    """wsm.execute with an explicit column width (the worker computes the first ``columns`` of its shard)."""
    version = leg["protocol_version"]
    t0 = clock()
    stream.sendall(wsm.EXECUTE.pack(wsm.PROTOCOL_MAGIC, version, 3, request_id, int(leg["layer"]), leg["n_embd"] * rows,
                                    len(payload), wsm.fnv32(payload), columns, rows) + payload)
    response = wsm.EXECUTE_RESPONSE.unpack(wsm.receive(stream, wsm.EXECUTE_RESPONSE.size))
    output = wsm.receive(stream, len(payload))
    t1 = clock()
    if response[:6] != (wsm.PROTOCOL_MAGIC, version, 4, 0, 0, request_id) or response[9] != wsm.fnv32(output):
        raise RuntimeError(f"EXECUTE {request_id} failed: {response[:10]}")
    return t0, t1, int(response[-1])


def drive_leg(port: int, config: Mapping[str, Any], leg: Mapping[str, Any], artifact_sha256: str, max_tokens: int,
              columns: int, out_path: Path, *, barrier: Callable[[], None] | None = None,
              clock: Callable[[], int] = time.monotonic_ns, sleep: Callable[[float], None] = time.sleep,
              connect: Callable = socket.create_connection, meta: Mapping[str, Any] | None = None) -> int:
    """One phone through idle-before, a segment per row count, idle-after and the closing call, at ``columns``
    width. With ``barrier`` (pair arms) every call waits at the barrier first, so the two phones' calls start
    together like the two partials of one split layer."""
    cadence, rows_list = config["cadence"], config.get("rows", [1, 2, 4])
    logical = 3 * leg["n_embd"] * leg["n_ff"] * 2 * columns // leg["n_ff"]
    sent = 0
    with out_path.open("w") as out, connect(("127.0.0.1", port), timeout=120) as stream:
        def emit(record: Mapping[str, Any]) -> None:
            out.write(json.dumps(record, sort_keys=True) + "\n")
            out.flush()
        stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        wsm.hello(stream, leg, artifact_sha256, max_tokens)
        emit({"kind": "meta", "schema": "ws20-phone-leg-v1", "op": "ffn", "device": leg["device"], "layer": leg["layer"],
              "n_embd": leg["n_embd"], "n_ff": leg["n_ff"], "columns": columns, "logical_bytes": logical,
              "paired": barrier is not None, **{k: cadence[k] for k in ("calls_per_token", "gap_call_ms", "gap_token_ms",
                                                                       "warmup_tokens")},
              "protocol_version": leg["protocol_version"], "transport": leg["transport"], **(meta or {})})
        t0 = clock()
        sleep(config["idle_s"])
        emit({"kind": "idle", "phase": "before", "t0_ns": t0, "t1_ns": clock()})
        payloads = {rows: wsm.synthetic_rows(leg["n_embd"], rows, 1000 + rows) for rows in rows_list}
        for rows in rows_list:
            tokens = wsm.phone_tokens(config, leg, rows)
            start, measured_start, measured = clock(), None, 0
            for token in range(-cadence["warmup_tokens"], tokens):
                if token == 0:
                    measured_start = clock()
                for position in range(cadence["calls_per_token"]):
                    if barrier is not None:
                        barrier()
                    sent += 1
                    c0, c1, compute_us = execute_width(stream, leg, sent, payloads[rows], rows, columns, clock)
                    emit({"kind": "call", "rows": rows, "warmup": token < 0, "token": token, "position": position,
                          "t_start_ns": c0, "t_end_ns": c1, "rpc_us": (c1 - c0) / 1e3, "compute_us": compute_us})
                    measured += token >= 0
                    sleep((cadence["gap_call_ms"] if position + 1 < cadence["calls_per_token"]
                           else cadence["gap_token_ms"]) / 1000.0)
            emit({"kind": "segment", "rows": rows, "t0_ns": measured_start if measured_start is not None else start,
                  "t1_ns": clock(), "calls": measured, "warmup_t0_ns": start})
        t0 = clock()
        sleep(config["idle_s"])
        emit({"kind": "idle", "phase": "after", "t0_ns": t0, "t1_ns": clock()})
        if barrier is not None:
            barrier()
        sent += 1   # the closing call exhausts the budget: the worker exits normally
        execute_width(stream, leg, sent, payloads[rows_list[0]], rows_list[0], columns, clock)
    return sent


def run_solo(config: Mapping[str, Any], arm: str, phone: Mapping[str, Any], directory: Path,
             session_factory: Callable = wsm.open_phone_session, **drive_options: Any) -> dict[str, Any]:
    leg = leg_for(config, phone, arm)
    session, worker = session_factory(config, leg)
    receipts: dict[str, Any] = {"preflight": session.preflight().to_json()}
    receipts["launch"] = session.start(directory / "worker.log").to_json()
    sent = 0
    try:
        sent = drive_leg(session.transport_contract().control_port, config, leg, worker.artifact_sha256,
                         worker.max_tokens, arm_width(config, arm), directory / "LEG.jsonl",
                         meta={"serial": worker.serial, "backend": worker.backend, "worker_path": worker.worker_path,
                               "shard_path": worker.shard_path, "arm": arm, "phone": phone["name"]}, **drive_options)
    finally:
        if session.active:
            receipts["stop"] = session.stop(served_calls=sent).to_json()
    receipts["calls_sent"] = sent
    return receipts


def run_pair(config: Mapping[str, Any], arm: str, directory: Path, session_factory: Callable = wsm.open_phone_session,
             **drive_options: Any) -> dict[str, Any]:
    """Both phones launched, then driven by two threads whose calls are released together by one barrier."""
    legs = [leg_for(config, phone, arm) for phone in config["phones"]]
    sessions = []
    receipts: dict[str, Any] = {"phones": {}}
    try:
        for leg in legs:
            session, worker = session_factory(config, leg)
            row = {"preflight": session.preflight().to_json()}
            row["launch"] = session.start(directory / (leg["name"].split(":")[-1] + "-worker.log")).to_json()
            receipts["phones"][leg["name"]] = row
            sessions.append((leg, session, worker))
        barrier = threading.Barrier(len(sessions))
        results: dict[str, Any] = {}

        def work(leg, session, worker):
            try:
                results[leg["name"]] = drive_leg(
                    session.transport_contract().control_port, config, leg, worker.artifact_sha256, worker.max_tokens,
                    arm_width(config, arm), directory / (leg["name"].split(":")[-1] + "-LEG.jsonl"),
                    barrier=barrier.wait, meta={"serial": worker.serial, "backend": worker.backend, "arm": arm,
                                                "phone": leg["name"].split(":")[-1]}, **drive_options)
            except BaseException as error:  # noqa: BLE001 - reported per phone; the barrier is broken for the other
                results[leg["name"]] = error
                barrier.abort()
        threads = [threading.Thread(target=work, args=row, daemon=True) for row in sessions]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for leg, _session, _worker in sessions:
            value = results.get(leg["name"])
            if isinstance(value, BaseException):
                raise RuntimeError(f"{leg['name']}: {type(value).__name__}: {value}") from value
            receipts["phones"][leg["name"]]["calls_sent"] = value
        return receipts
    finally:
        for leg, session, _worker in sessions:
            if session.active:
                sent = results.get(leg["name"]) if "results" in locals() else None
                receipts["phones"][leg["name"]]["stop"] = session.stop(
                    served_calls=sent if isinstance(sent, int) else 0).to_json()


def run(config: Mapping[str, Any], output: Path, *, only: set[str] | None = None,
        sampler_factory: Callable = wsm.default_sampler, solo_runner: Callable = run_solo,
        pair_runner: Callable = run_pair, sleep: Callable[[float], None] = time.sleep) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=False)
    wsm.write_json(output / "CONFIG.json", config)
    wsm.write_json(output / "PLAN.json", plan(config))
    sampler = (sampler_factory(config.get("host", {}).get("gpu_index"))
               if sampler_factory is wsm.default_sampler else sampler_factory())
    sampler.start()
    results: list[dict[str, Any]] = []
    try:
        for arm in config.get("arms", list(ARMS)):
            units = [(arm, None)] if arm in PAIR_ARMS else [(arm, phone) for phone in config["phones"]]
            for arm_name, phone in units:
                name = arm_name if phone is None else f"{arm_name}:{phone['name']}"
                if only and name not in only and arm_name not in only:
                    continue
                directory = output / name.replace(":", "_")
                directory.mkdir()
                entry: dict[str, Any] = {"name": name, "arm": arm_name, "status": "FAILED"}
                try:
                    entry["receipts"] = pair_runner(config, arm_name, directory) if phone is None \
                        else solo_runner(config, arm_name, phone, directory)
                    entry["status"] = "PASS"
                except Exception as error:  # noqa: BLE001 - one unit's failure is recorded; the others still run
                    entry["error"] = f"{type(error).__name__}: {error}"
                results.append(entry)
                wsm.write_json(output / "RUN_RESULT.json", results)
                sleep(config.get("cooldown_s", 10))
    finally:
        sampler.stop()
        with (output / "HOST_SAMPLES.jsonl").open("w") as stream:
            for row in sampler.rows():
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    return results


# ---------------------------------------------------------------- analysis

def _segments(records: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    out = {}
    for segment in (r for r in records if r["kind"] == "segment"):
        calls = [r for r in records if r["kind"] == "call" and r["rows"] == segment["rows"] and not r["warmup"]]
        out[segment["rows"]] = {"t0_ns": segment["t0_ns"], "t1_ns": segment["t1_ns"], "calls": len(calls),
                                "rpc_ms": statistics.median(c["rpc_us"] for c in calls) / 1000 if calls else None,
                                "rpc_ms_mean": statistics.fmean(c["rpc_us"] for c in calls) / 1000 if calls else None,
                                "compute_ms": statistics.median(c["compute_us"] for c in calls) / 1000 if calls else None,
                                "starts_ns": [c["t_start_ns"] for c in calls], "ends_ns": [c["t_end_ns"] for c in calls]}
    return out


def _phone_j_per_call(directory: Path, records, meter, logs, device: str) -> dict[int, float | None]:
    """Incremental energy per call of the phone's own domain, WS16 convention (idle windows of this leg)."""
    idles = [r for r in records if r["kind"] == "idle"]
    windows = [wsm.window_energy([], meter, logs, r["t0_ns"], r["t1_ns"]) for r in idles]
    seconds = sum(w["seconds"] for w in windows)
    if not seconds or any(w["j"].get(device) is None for w in windows):
        return {s["rows"]: None for s in records if s["kind"] == "segment"}
    idle_w = sum(w["j"][device] for w in windows) / seconds
    out = {}
    for segment in (r for r in records if r["kind"] == "segment"):
        window = wsm.window_energy([], meter, logs, segment["t0_ns"], segment["t1_ns"])
        value = window["j"].get(device)
        out[segment["rows"]] = None if value is None or not segment["calls"] else \
            (value - idle_w * window["seconds"]) / segment["calls"]
    return out


def analyze(output: Path, meter_path: Path | None = None, *, max_degradation: float | None = None,
            max_penalty_ms: float | None = None) -> dict[str, Any]:
    config = json.loads((output / "CONFIG.json").read_text())
    meter, logs = wsm.load_meter(meter_path)
    rows_list = config.get("rows", [1, 2, 4])
    max_degradation = float(config.get("max_degradation", 1.10) if max_degradation is None else max_degradation)
    max_penalty_ms = float(config.get("max_penalty_ms", 1.0) if max_penalty_ms is None else max_penalty_ms)
    legs: dict[str, dict[str, Any]] = {}      # "arm:phone" -> {rows -> segment stats}, energy
    for directory in sorted(p for p in output.iterdir() if p.is_dir()):
        for leg_path in sorted(directory.glob("*LEG.jsonl")):
            records = wsm.read_jsonl(leg_path)
            meta = next(r for r in records if r["kind"] == "meta")
            name = f"{meta['arm']}:{meta['phone']}"
            legs[name] = {"segments": _segments(records), "columns": meta["columns"], "device": meta["device"],
                          "logical_bytes": meta["logical_bytes"],
                          "j_per_call": _phone_j_per_call(directory, records, meter, logs, meta["device"])}
    phones = [phone["name"] for phone in config["phones"]]
    table = []
    worst_degradation, worst_penalty = 0.0, 0.0
    flags: list[str] = []
    for rows in rows_list:
        row: dict[str, Any] = {"rows": rows, "phones": {}}
        pair_rpcs = []
        for phone in phones:
            cell: dict[str, Any] = {}
            for arm in ARMS:
                leg = legs.get(f"{arm}:{phone}")
                seg = leg["segments"].get(rows) if leg else None
                if seg:
                    cell[arm] = {"rpc_ms": seg["rpc_ms"], "compute_ms": seg["compute_ms"], "calls": seg["calls"],
                                 "j_per_call": leg["j_per_call"].get(rows)}
            for arm_pair, arm_solo in (("pair-half", "solo-half"), ("pair-full", "solo-full")):
                if arm_pair in cell and arm_solo in cell and cell[arm_solo]["compute_ms"]:
                    cell[arm_pair]["degradation_compute"] = cell[arm_pair]["compute_ms"] / cell[arm_solo]["compute_ms"]
                    cell[arm_pair]["degradation_rpc"] = cell[arm_pair]["rpc_ms"] / cell[arm_solo]["rpc_ms"]
                    if arm_pair == "pair-half":
                        worst_degradation = max(worst_degradation, cell[arm_pair]["degradation_compute"])
            if "pair-half" in cell:
                pair_rpcs.append(cell["pair-half"]["rpc_ms"])
            row["phones"][phone] = cell
        halves = [row["phones"][p]["solo-half"]["rpc_ms"] for p in phones
                  if "solo-half" in row["phones"][p] and row["phones"][p]["solo-half"]["rpc_ms"] is not None]
        if len(pair_rpcs) == 2 and len(halves) == 2:
            # the pair time a split layer would wait for, against what the slower solo half-call alone costs
            pair_time = max(pair_rpcs)
            row["pair_ms"] = pair_time
            row["slower_solo_half_ms"] = max(halves)
            row["penalty_ms"] = pair_time - max(halves)
            worst_penalty = max(worst_penalty, row["penalty_ms"])
            # what the planner compares against: the slower full-width solo call
            fulls = [row["phones"][p]["solo-full"]["rpc_ms"] for p in phones if "solo-full" in row["phones"][p]]
            if len(fulls) == 2 and all(fulls):
                row["speedup_vs_single_phone"] = min(fulls) / pair_time
        table.append(row)
    if not any("penalty_ms" in row for row in table):
        flags.append("PAIR_ARMS_MISSING")
    status = "PASS" if not flags and worst_degradation <= max_degradation and worst_penalty <= max_penalty_ms else "FAIL"
    if worst_degradation > max_degradation:
        flags.append("DEGRADATION_ABOVE_LIMIT")
    if worst_penalty > max_penalty_ms:
        flags.append("PENALTY_ABOVE_LIMIT")
    return {"schema": RESULT_SCHEMA, "status": status, "flags": flags, "output": str(output),
            "meter": str(meter_path) if meter_path else None,
            "degradation": round(max(1.0, worst_degradation), 4), "penalty_ms": round(max(0.0, worst_penalty), 4),
            "limits": {"max_degradation": max_degradation, "max_penalty_ms": max_penalty_ms},
            "convention": "degradation = concurrent compute / solo compute at the same width (worst phone, worst rows); "
                          "penalty_ms = max concurrent half-call RPC - max solo half-call RPC (worst rows); "
                          "j_per_call = phone energy above its own idle per call (WS16 convention)",
            "rows": table}


def markdown(result: Mapping[str, Any]) -> str:
    lines = [f"# WS20 parallel-split concurrency: **{result['status']}** (degradation x{result['degradation']}, "
             f"penalty {result['penalty_ms']:.2f} ms; limits x{result['limits']['max_degradation']}, "
             f"{result['limits']['max_penalty_ms']} ms)", ""]
    if result["flags"]:
        lines.append("flags: " + ", ".join(result["flags"]))
        lines.append("")
    lines += ["| rows | phone | solo full rpc/compute ms | solo half rpc/compute | pair half rpc/compute (degr.) | "
              "pair full rpc/compute (degr.) | J/call solo-half / pair-half |", "|---|---|---|---|---|---|---|"]

    def fmt(cell, arm):
        if arm not in cell:
            return "-"
        c = cell[arm]
        text = f"{c['rpc_ms']:.2f} / {c['compute_ms']:.2f}" if c["rpc_ms"] is not None else "-"
        if "degradation_compute" in c:
            text += f" (x{c['degradation_compute']:.3f})"
        return text

    def energy(cell):
        parts = []
        for arm in ("solo-half", "pair-half"):
            value = cell.get(arm, {}).get("j_per_call")
            parts.append("-" if value is None else f"{value:.3f}")
        return " / ".join(parts)

    for row in result["rows"]:
        for phone, cell in row["phones"].items():
            lines.append(f"| {row['rows']} | {phone} | {fmt(cell, 'solo-full')} | {fmt(cell, 'solo-half')} | "
                         f"{fmt(cell, 'pair-half')} | {fmt(cell, 'pair-full')} | {energy(cell)} |")
        if "pair_ms" in row:
            lines.append(f"| {row['rows']} | pair | pair time {row['pair_ms']:.2f} ms vs slower solo half "
                         f"{row['slower_solo_half_ms']:.2f} ms: penalty {row['penalty_ms']:.2f} ms"
                         + (f"; speed-up vs the faster single phone x{row['speedup_vs_single_phone']:.2f}"
                            if "speedup_vs_single_phone" in row else "") + " | | | | | |")
    return "\n".join(lines)


# ---------------------------------------------------------------- smoke / CLI

def smoke_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """A 2-token, 1-row, 2 s-idle copy of the two solo-full legs and the pair-half arm."""
    small = json.loads(json.dumps(config))
    small.update(rows=[1], idle_s=2, segment_seconds=1, cooldown_s=1, arms=["solo-full", "pair-half"])
    small["cadence"] = dict(small["cadence"], warmup_tokens=1)
    return small


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan"); p.add_argument("config", type=Path)
    p = sub.add_parser("smoke"); p.add_argument("config", type=Path); p.add_argument("output", type=Path)
    p = sub.add_parser("run"); p.add_argument("config", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--only")
    p = sub.add_parser("analyze"); p.add_argument("output", type=Path); p.add_argument("--meter", type=Path)
    p.add_argument("--json", type=Path); p.add_argument("--md", type=Path)
    p.add_argument("--max-degradation", type=float); p.add_argument("--max-penalty-ms", type=float)
    args = parser.parse_args(argv)
    if args.command == "plan":
        print(json.dumps(plan(load_config(args.config)), indent=2))
    elif args.command == "smoke":
        results = run(smoke_config(load_config(args.config)), args.output)
        print(json.dumps(results, indent=2, default=str))
        return 0 if all(r["status"] == "PASS" for r in results) else 1
    elif args.command == "run":
        results = run(load_config(args.config), args.output, only=set(args.only.split(",")) if args.only else None)
        print(json.dumps([{k: r[k] for k in ("name", "status", "error") if k in r} for r in results], indent=2))
        return 0 if all(r["status"] == "PASS" for r in results) else 1
    else:
        result = analyze(args.output, args.meter, max_degradation=args.max_degradation, max_penalty_ms=args.max_penalty_ms)
        if args.json:
            wsm.write_json(args.json, result)
        if args.md:
            args.md.write_text(markdown(result) + "\n")
        print(markdown(result))
        return 0 if result["status"] == "PASS" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
