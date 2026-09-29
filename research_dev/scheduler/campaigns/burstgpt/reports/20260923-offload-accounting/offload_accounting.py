#!/usr/bin/env python3
"""Account for where the phone offload went in a phone-assisted trace run (and its desktop baseline).

Inputs per arm: RESULT.json, ADAPTIVE_DECODE_OBSERVATIONS.json (treatment), resource-samples.jsonl and
the server_logs_<arm>.json written by parse_server_logs.py. Everything here is read from the recorded
artifacts; nothing is executed on hardware.

Token attribution rule (from coverage_energy_data.py): a request's phone-assisted decode tokens are
its phone calls per released layer averaged over layers (physical_execution_proof.phone_calls_by_layer).
Window token counts come from the adaptive decode windows (token_start/token_end) and are labelled by
the window's policy (baseline or phone split fraction), role (exploration/exploitation), active batch,
and, for batch-2 baseline windows, by whether the co-tenant slot ran a phone policy at the same time
(mixed policies cannot be batched by the server, so both slots decode in alternating forward passes).

Power states: each resource sample interval is labelled from the request timelines (idle, model load,
prefill, decode by model/batch/policy mix) and the RAPL package energy delta plus NVML board power
are summed per state.

    python3 offload_accounting.py --treatment <run> --baseline <run> \
        --treatment-logs server_logs_treatment.json --baseline-logs server_logs_baseline.json --out-dir <dir>
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import statistics

TRACE = "burstgpt_longtail_v1"


def model_key(model_id: str) -> str:
    for prefix in ("qwen", "gemma", "llama"):
        if model_id.startswith(prefix):
            return prefix
    return model_id.split("-")[0]


def short(request_id: str) -> str:
    return request_id.split(":")[-1]


# ----------------------------------------------------------------------------------------- requests

def request_rows(result: dict, logs: dict) -> list[dict]:
    paid_start_ns = result["paid_start_ns"]
    timings = {}
    for proc in logs["processes"]:
        for task, entry in proc["timings"].items():
            if "prompt_tokens" in entry and "eval_tokens" in entry:
                timings.setdefault((entry["prompt_tokens"], entry["eval_tokens"]), []).append(
                    {**entry, "process": proc["index"], "role": proc["role"]})
    rows = []
    for row in result["request_results"]:
        proof = row.get("physical_execution_proof") or {}
        by_layer = [entry["calls"] for entry in proof.get("phone_calls_by_layer") or []
                    if isinstance(entry, dict) and "calls" in entry]
        dispatch = row["dispatch_receipts"][-1] if row["dispatch_receipts"] else {}
        first_token_s = (row["first_token_ns"] - paid_start_ns) / 1e9 if row.get("first_token_ns") else None
        end_s = row["completion"]["actual_end_us"] / 1e6
        sched_s = dispatch.get("scheduled_start_us", row["replay_arrival_us"]) / 1e6
        timing = (timings.get((row["input_tokens"], row["output_tokens"])) or [None])[0]
        decode_s = (end_s - first_token_s) if first_token_s is not None else None
        rows.append({
            "request_id": short(row["request_id"]), "model": model_key(row["model_id"]),
            "arrival_s": row["replay_arrival_us"] / 1e6, "scheduled_start_s": sched_s,
            "queue_wait_s": round(sched_s - row["replay_arrival_us"] / 1e6, 3),
            "first_token_s": first_token_s, "end_s": end_s, "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"], "latency_s": row["actual_latency_us"] / 1e6,
            "time_to_first_token_s": round(first_token_s - sched_s, 3) if first_token_s is not None else None,
            "decode_s": round(decode_s, 3) if decode_s is not None else None,
            "decode_ms_per_token": round(1e3 * decode_s / max(1, row["output_tokens"] - 1), 1)
            if decode_s is not None and row["output_tokens"] > 1 else None,
            "server_prompt_ms": timing["prompt_ms"] if timing else None,
            "server_eval_ms_per_token": round(timing["eval_ms"] / max(1, timing["eval_tokens"]), 1) if timing else None,
            "phone_calls": proof.get("phone_call_count", 0), "released_layers": len(by_layer),
            "assisted_tokens": round(sum(by_layer) / len(by_layer)) if by_layer else 0,
            "attempts": len(row["attempt_ticket_ids"]), "executor": row["actual_executor_id"],
        })
    rows.sort(key=lambda row: row["arrival_s"])
    # concurrency partners: other large requests whose decode overlaps this request's decode
    for row in rows:
        partners = []
        for other in rows:
            if other is row or other["first_token_s"] is None or row["first_token_s"] is None:
                continue
            lo = max(row["first_token_s"], other["first_token_s"])
            hi = min(row["end_s"], other["end_s"])
            if hi - lo > 1.0:
                partners.append(other["request_id"])
        row["decode_partners"] = ",".join(partners)
    return rows


# ------------------------------------------------------------------------------------------ windows

def longtail_groups(run: pathlib.Path) -> list[dict]:
    store = run / "ADAPTIVE_DECODE_OBSERVATIONS.json"
    if not store.exists():
        return []
    return [group for group in json.loads(store.read_text())["groups"] if group["request_id"].startswith(TRACE)]


def window_rows(groups: list[dict], model_of: dict[str, str]) -> list[dict]:
    rows = []
    for group in groups:
        request_id = group["request_id"]
        for window in group["windows"]:
            policy = window["policy"]
            rows.append({
                "request_id": short(request_id), "model": model_of[request_id],
                "token_start": window["token_start"], "token_end": window["token_end"],
                "tokens": window["token_end"] - window["token_start"],
                "started_s": window["started_at_us"] / 1e6, "finished_s": window["finished_at_us"] / 1e6,
                "policy": "baseline" if policy["baseline"] else "phone",
                "fraction_ppm": policy["split_fraction_ppm"], "role": window.get("window_role"),
                "active_batch": window.get("active_batch"), "slot": window.get("slot_id"),
                "latency_ms_per_token": (window.get("latency_per_token_us") or 0) / 1e3,
                "measurement_eligible": window.get("measurement_eligible", True),
                "energy_kind": window.get("energy_attribution_kind"),
                "cpu_uj": (window.get("fleet_energy_uj_by_domain") or {}).get("cpu-package", 0),
                "gpu_uj": (window.get("fleet_energy_uj_by_domain") or {}).get("gpu-board", 0),
                "phone_calls": window.get("completed_phone_calls", 0),
                "phone_rpc_us": window.get("rpc_us", 0), "phone_compute_us": window.get("phone_compute_us", 0),
                "failure_reason": window.get("failure_reason"),
            })
    rows.sort(key=lambda row: (row["request_id"], row["token_start"]))
    # co-tenant policy at the window midpoint: which policy the other decoding slot ran
    by_request = collections.defaultdict(list)
    for row in rows:
        by_request[row["request_id"]].append(row)
    for row in rows:
        mid = (row["started_s"] + row["finished_s"]) / 2
        others = set()
        for request_id, lst in by_request.items():
            if request_id == row["request_id"]:
                continue
            for other in lst:
                if other["started_s"] <= mid <= other["finished_s"]:
                    others.add(other["policy"])
        row["cotenant_policy"] = "+".join(sorted(others)) if others else "none"
        if row["active_batch"] and row["active_batch"] >= 2:
            if row["policy"] == "phone":
                row["category"] = "phone_b2_mixed" if "baseline" in others or not others else "phone_b2_coherent"
            else:
                row["category"] = "baseline_b2_mixed" if "phone" in others else "baseline_b2_both_baseline"
        elif row["policy"] == "phone":
            row["category"] = "phone_b1_" + ("exploit" if row["role"] == "exploitation" else "probe")
        else:
            row["category"] = "baseline_b1_" + ("exploit" if row["role"] == "exploitation" else "probe")
    return rows


def token_attribution(result: dict, windows: list[dict], requests: list[dict]) -> dict:
    """Decode tokens of the trace by (model, category)."""
    out_by_request = {row["request_id"]: row["output_tokens"] for row in requests}
    model_by_request = {row["request_id"]: row["model"] for row in requests}
    table = collections.defaultdict(lambda: collections.Counter())
    covered = collections.Counter()
    for row in windows:
        table[row["model"]][row["category"]] += row["tokens"]
        covered[row["request_id"]] += row["tokens"]
    for request_id, output in out_by_request.items():
        model = model_by_request[request_id]
        if model == "llama":
            table[model]["no_phone_candidates"] += output
        else:
            table[model]["uncovered_first_token_and_tail"] += output - covered.get(request_id, 0)
    totals = collections.Counter()
    for model, counter in table.items():
        for category, tokens in counter.items():
            totals[category] += tokens
    table["all"] = totals
    payload = {}
    for model, counter in table.items():
        total = sum(counter.values())
        phone = sum(v for k, v in counter.items() if k.startswith("phone_"))
        payload[model] = {"tokens": total, "phone_policy_tokens": phone,
                          "phone_policy_share": round(phone / total, 4) if total else None,
                          "by_category": dict(sorted(counter.items()))}
    # cross-check against the proof-derived assisted tokens
    proof = collections.Counter()
    for row in requests:
        proof[row["model"]] += row["assisted_tokens"]
    proof["all"] = sum(proof.values())
    for model in payload:
        payload[model]["proof_assisted_tokens"] = proof.get(model, 0)
    return payload


def window_latency_energy(windows: list[dict]) -> list[dict]:
    """Measured ms/token and J/token by (model, batch, policy, co-tenant mix), measurement-eligible windows."""
    groups = collections.defaultdict(list)
    for row in windows:
        if not row["measurement_eligible"] or row["tokens"] <= 0:
            continue
        key = (row["model"], row["active_batch"], row["policy"],
               row["fraction_ppm"] if row["policy"] == "phone" else 0, row["cotenant_policy"])
        groups[key].append(row)
    rows = []
    for key, lst in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0, kv[0][2], -kv[0][3], kv[0][4])):
        tokens = sum(r["tokens"] for r in lst)
        seconds = sum(r["finished_s"] - r["started_s"] for r in lst)
        # window energy is the whole-fleet host energy (RAPL + NVML) over the window; at batch 1 only
        # this request runs, so it is attributable. At batch 2 it is shared by both slots, so the
        # per-token figure below is per token of THIS slot and must be halved for a per-pass view.
        cpu = sum(r["cpu_uj"] for r in lst) / 1e6
        gpu = sum(r["gpu_uj"] for r in lst) / 1e6
        rows.append({
            "model": key[0], "active_batch": key[1], "policy": key[2], "fraction_ppm": key[3],
            "cotenant_policy": key[4], "windows": len(lst), "tokens": tokens,
            "ms_per_token": round(1e3 * seconds / tokens, 1),
            "host_j_per_slot_token": round((cpu + gpu) / tokens, 2),
            "host_j_per_pass_token": round((cpu + gpu) / tokens / (key[1] or 1), 2),
            "cpu_w": round(cpu / seconds, 1) if seconds else None,
            "gpu_w": round(gpu / seconds, 1) if seconds else None,
            "energy_kinds": ",".join(sorted({r["energy_kind"] or "none" for r in lst})),
        })
    return rows


def request_window_summary(groups: list[dict], windows: list[dict]) -> list[dict]:
    per = collections.defaultdict(lambda: collections.Counter())
    for row in windows:
        per[row["request_id"]][row["category"]] += row["tokens"]
        per[row["request_id"]]["_windows"] += 1
    rows = []
    for group in groups:
        rid = short(group["request_id"])
        counter = per[rid]
        rows.append({"request_id": rid, "state_history": ">".join(group["state_history"]),
                     "terminal_status": group["terminal_status"], "terminal_reason": group.get("terminal_reason"),
                     "unmeasured_tail_tokens": group.get("unmeasured_tail_tokens"),
                     "unmeasured_tail_reason": group.get("unmeasured_tail_reason"),
                     "windows": counter.pop("_windows", 0), **{k: v for k, v in sorted(counter.items())}})
    rows.sort(key=lambda row: row["request_id"])
    return rows


# --------------------------------------------------------------------------------- phone utilization

def phone_utilization(result: dict, requests: list[dict], windows: list[dict], logs: dict) -> dict:
    """Phone busy time per model (from window rpc time and from FFNSHAPE call statistics) and idle time
    of each model's resident layers while the desktop was serving the other model."""
    rpc_by_model = collections.Counter()
    compute_by_model = collections.Counter()
    for row in windows:
        rpc_by_model[row["model"]] += row["phone_rpc_us"] / 1e6
        compute_by_model[row["model"]] += row["phone_compute_us"] / 1e6
    shape_calls = collections.defaultdict(lambda: {"calls": 0, "rpc_s": 0.0, "compute_s": 0.0})
    for proc in logs["processes"]:
        model = "qwen" if proc["role"] == "hot" else ("gemma" if proc["role"] == "cold" else "llama")
        for shape in proc["shapes"]:
            entry = shape_calls[(model, shape["columns"])]
            entry["calls"] += shape["calls"]
            entry["rpc_s"] += shape["calls"] * shape["rpc_mean_ms"] / 1e3
            entry["compute_s"] += shape["calls"] * shape["compute_mean_ms"] / 1e3
    meta = result["trace_energy"]["estimation_metadata"]
    duration_s = result["duration_us"] / 1e6
    # desktop model phases from server processes
    phases = []
    for proc in logs["processes"]:
        if proc["role"] in ("hot", "cold"):
            phases.append({"model": "qwen" if proc["role"] == "hot" else "gemma", "start_s": proc["start_s"],
                           "end_s": proc["end_s"], "load_s": proc["loaded_after_s"]})
    phase_time = collections.Counter()
    for phase in phases:
        phase_time[phase["model"]] += max(0.0, phase["end_s"] - phase["start_s"])
    # decode time per model where phone policy was active (union of phone windows)
    phone_active = collections.Counter()
    decode_time = collections.Counter()
    for row in requests:
        if row["first_token_s"] is not None and row["model"] != "llama":
            decode_time[row["model"]] += row["end_s"] - row["first_token_s"]
    for model in ("qwen", "gemma"):
        intervals = sorted((r["started_s"], r["finished_s"]) for r in windows
                           if r["model"] == model and r["policy"] == "phone")
        merged = []
        for lo, hi in intervals:
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        phone_active[model] = sum(hi - lo for lo, hi in merged)
    resident = {"qwen": 6417137664, "gemma": 2831056896}
    payload = {
        "duration_s": duration_s,
        "phone_active_time_s_result": (meta.get("phone_active_time_ns") or 0) / 1e9,
        "by_model": {},
        "shape_calls": [{"model": k[0], "columns": k[1], **v} for k, v in sorted(shape_calls.items())],
        "desktop_phases": phases,
    }
    for model in ("qwen", "gemma"):
        busy = sum(v["rpc_s"] for k, v in shape_calls.items() if k[0] == model)
        payload["by_model"][model] = {
            "phone_rpc_busy_s_from_shapes": round(busy, 1),
            "phone_rpc_busy_s_from_windows": round(rpc_by_model[model], 1),
            "phone_compute_s_from_windows": round(compute_by_model[model], 1),
            "busy_share_of_run": round(busy / duration_s, 4),
            "desktop_phase_s": round(phase_time[model], 1),
            "other_model_phase_s": round(sum(v for k, v in phase_time.items() if k != model), 1),
            "decode_s": round(decode_time[model], 1),
            "phone_policy_active_s": round(phone_active[model], 1),
            "decode_s_without_phone_policy": round(decode_time[model] - phone_active[model], 1),
            "resident_bytes": resident[model],
            "idle_share_other_model_resident": round(sum(v for k, v in phase_time.items() if k != model) / duration_s, 4),
        }
    weighted = sum(resident[m] * phase_time[m] for m in resident) / (sum(resident.values()) * duration_s)
    payload["resident_bytes_useful_time_weighted_share"] = round(weighted, 4)
    return payload


# --------------------------------------------------------------------------------------------- queue

def queue_analysis(requests: list[dict]) -> dict:
    large = [row for row in requests if row["model"] != "llama"]
    waits = [row["queue_wait_s"] for row in large]
    by_model = {}
    for model in ("qwen", "gemma", "llama"):
        rows = [row for row in requests if row["model"] == model]
        by_model[model] = {"requests": len(rows),
                           "queue_wait_s_sum": round(sum(r["queue_wait_s"] for r in rows), 1),
                           "queue_wait_s_mean": round(statistics.mean(r["queue_wait_s"] for r in rows), 1),
                           "queue_wait_s_max": round(max(r["queue_wait_s"] for r in rows), 1)}
    # model switches in service order
    order = sorted(large, key=lambda row: row["scheduled_start_s"])
    switches = sum(1 for a, b in zip(order, order[1:]) if a["model"] != b["model"])
    # queue depth by model over time at each scheduling instant
    depth_samples = []
    for row in order:
        t = row["scheduled_start_s"]
        queued = collections.Counter(r["model"] for r in large if r["arrival_s"] <= t and r["scheduled_start_s"] > t)
        depth_samples.append({"t_s": t, "starting": row["request_id"], "model": row["model"],
                              "queued_qwen": queued["qwen"], "queued_gemma": queued["gemma"]})
    concurrent = sum(1 for row in large if row["decode_partners"])
    return {"large_requests": len(large), "queue_wait_s_sum": round(sum(waits), 1),
            "queue_wait_s_mean": round(statistics.mean(waits), 1), "queue_wait_s_max": round(max(waits), 1),
            "by_model": by_model, "model_switches_in_service_order": switches,
            "requests_with_decode_partner": concurrent, "depth_at_dispatch": depth_samples}


# ---------------------------------------------------------------------------------- release/restore

def release_restore(logs: dict) -> dict:
    events = [dict(e, process=proc["index"], role=proc["role"]) for proc in logs["processes"] for e in proc["dormant"]]
    events.sort(key=lambda e: e["t_s"])
    by_kind = collections.defaultdict(list)
    for event in events:
        by_kind[(event["role"], event["kind"])].append(event)
    summary = {}
    for key, lst in sorted(by_kind.items()):
        elapsed = [e["elapsed_us"] / 1e3 for e in lst]
        summary[f"{key[0]}:{key[1]}"] = {"count": len(lst), "elapsed_ms_sum": round(sum(elapsed), 1),
                                         "elapsed_ms_mean": round(statistics.mean(elapsed), 1),
                                         "elapsed_ms_max": round(max(elapsed), 1),
                                         "bytes_mean_gb": round(statistics.mean(e["bytes"] for e in lst) / 1e9, 2)}
    mixed = sum(max([c for _, c in proc["mixed_skips"]], default=0) for proc in logs["processes"])
    return {"events": events, "summary": summary, "total_elapsed_s": round(sum(e["elapsed_us"] for e in events) / 1e6, 2),
            "releases": sum(1 for e in events if e["kind"] == "release"),
            "restores": sum(1 for e in events if e["kind"] == "restore"),
            "mixed_policy_release_skips": mixed}


# --------------------------------------------------------------------------------------- power states

def load_samples(run: pathlib.Path, paid_start_ns: int) -> list[tuple[float, float, float]]:
    """(t_s, cpu_energy_j_delta, gpu_j) per sample interval, t at interval end."""
    rows = [json.loads(line) for line in (run / "resource-samples.jsonl").read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: row["t_ns"])
    out = []
    for before, after in zip(rows, rows[1:]):
        seconds = (after["t_ns"] - before["t_ns"]) / 1e9
        if seconds <= 0:
            continue
        delta = after["rapl_package"]["energy_uj"] - before["rapl_package"]["energy_uj"]
        if delta < 0:
            delta += before["rapl_package"].get("max_energy_range_uj", 0)
        cpu_j = max(0.0, delta / 1e6)
        gpu_j = after["gpu"]["power_mw"] / 1e3 * seconds
        out.append(((before["t_ns"] - paid_start_ns) / 1e9, (after["t_ns"] - paid_start_ns) / 1e9, cpu_j, gpu_j))
    return out


def power_states(result: dict, requests: list[dict], windows: list[dict], logs: dict, run: pathlib.Path) -> dict:
    samples = load_samples(run, result["paid_start_ns"])
    loads = [(p["start_s"], p["start_s"] + (p["loaded_after_s"] or 0), "qwen" if p["role"] == "hot" else "gemma" if p["role"] == "cold" else "llama")
             for p in logs["processes"]]
    large = [row for row in requests if row["first_token_s"] is not None]
    win_by_request = collections.defaultdict(list)
    for row in windows:
        win_by_request[row["request_id"]].append(row)

    def state_at(t: float) -> str:
        decoding = [row for row in large if row["first_token_s"] <= t < row["end_s"]]
        if decoding:
            models = {row["model"] for row in decoding}
            model = "+".join(sorted(models))
            policies = []
            for row in decoding:
                policy = "unknown"
                for win in win_by_request.get(row["request_id"], ()):
                    if win["started_s"] <= t < win["finished_s"]:
                        policy = win["policy"]
                        break
                if row["model"] == "llama":
                    policy = "baseline"
                policies.append(policy)
            mix = "+".join(sorted(set(policies)))
            return f"decode:{model}:b{len(decoding)}:{mix}"
        prefill = [row for row in large if row["scheduled_start_s"] <= t < row["first_token_s"]]
        for lo, hi, model in loads:
            if lo <= t < hi:
                return f"load:{model}"
        if prefill:
            return "prefill:" + "+".join(sorted({row["model"] for row in prefill}))
        return "idle"

    agg = collections.defaultdict(lambda: {"seconds": 0.0, "cpu_j": 0.0, "gpu_j": 0.0})
    for lo, hi, cpu_j, gpu_j in samples:
        entry = agg[state_at((lo + hi) / 2)]
        entry["seconds"] += hi - lo
        entry["cpu_j"] += cpu_j
        entry["gpu_j"] += gpu_j
    rows = []
    for state, entry in sorted(agg.items(), key=lambda kv: -kv[1]["seconds"]):
        rows.append({"state": state, "seconds": round(entry["seconds"], 1),
                     "cpu_kj": round(entry["cpu_j"] / 1e3, 2), "gpu_kj": round(entry["gpu_j"] / 1e3, 2),
                     "host_kj": round((entry["cpu_j"] + entry["gpu_j"]) / 1e3, 2),
                     "cpu_w": round(entry["cpu_j"] / entry["seconds"], 1) if entry["seconds"] else None,
                     "gpu_w": round(entry["gpu_j"] / entry["seconds"], 1) if entry["seconds"] else None,
                     "host_w": round((entry["cpu_j"] + entry["gpu_j"]) / entry["seconds"], 1) if entry["seconds"] else None})
    total_s = sum(r["seconds"] for r in rows)
    total_kj = sum(r["host_kj"] for r in rows)
    coarse = collections.defaultdict(lambda: {"seconds": 0.0, "host_kj": 0.0, "cpu_kj": 0.0, "gpu_kj": 0.0})
    for r in rows:
        key = r["state"].split(":")[0]
        if key == "decode":
            key = "decode:" + r["state"].split(":")[3]
        coarse[key]["seconds"] += r["seconds"]
        coarse[key]["host_kj"] += r["host_kj"]
        coarse[key]["cpu_kj"] += r["cpu_kj"]
        coarse[key]["gpu_kj"] += r["gpu_kj"]
    for entry in coarse.values():
        entry["host_w"] = round(entry["host_kj"] * 1e3 / entry["seconds"], 1) if entry["seconds"] else None
        for k in ("seconds", "host_kj", "cpu_kj", "gpu_kj"):
            entry[k] = round(entry[k], 2)
    idle = agg.get("idle", {"seconds": 0, "cpu_j": 0, "gpu_j": 0})
    floor_w = (idle["cpu_j"] + idle["gpu_j"]) / idle["seconds"] if idle["seconds"] else None
    return {"states": rows, "coarse": dict(coarse), "sampled_seconds": round(total_s, 1),
            "sampled_host_kj": round(total_kj, 2),
            "idle_floor_w": round(floor_w, 1) if floor_w else None,
            "idle_floor_cpu_w": round(idle["cpu_j"] / idle["seconds"], 1) if idle["seconds"] else None,
            "idle_floor_gpu_w": round(idle["gpu_j"] / idle["seconds"], 1) if idle["seconds"] else None,
            "floor_energy_over_run_kj": round(floor_w * total_s / 1e3, 1) if floor_w else None}


# ----------------------------------------------------------------------------------------- anomalies

def anomalies(requests: list[dict], logs: dict, arm: str) -> list[dict]:
    out = []
    by_model = collections.defaultdict(list)
    for row in requests:
        if row["decode_ms_per_token"] and row["model"] != "llama" and not row["decode_partners"]:
            by_model[row["model"]].append(row["decode_ms_per_token"])
    median = {m: statistics.median(v) for m, v in by_model.items() if v}
    for row in requests:
        if row["decode_ms_per_token"] and row["model"] in median and row["decode_ms_per_token"] > 1.25 * median[row["model"]]:
            out.append({"arm": arm, "kind": "slow_decode", "request_id": row["request_id"], "model": row["model"],
                        "decode_ms_per_token": row["decode_ms_per_token"], "solo_median_ms_per_token": round(median[row["model"]], 1),
                        "partners": row["decode_partners"]})
        if row["time_to_first_token_s"] and row["time_to_first_token_s"] > 90:
            out.append({"arm": arm, "kind": "long_time_to_first_token", "request_id": row["request_id"], "model": row["model"],
                        "time_to_first_token_s": row["time_to_first_token_s"], "input_tokens": row["input_tokens"]})
    for proc in logs["processes"]:
        if proc["role"] in ("hot", "cold") and (proc["loaded_after_s"] or 0) > 60:
            out.append({"arm": arm, "kind": "slow_model_load", "process": proc["index"], "role": proc["role"],
                        "load_s": proc["loaded_after_s"], "start_s": proc["start_s"]})
    order = sorted((r for r in requests if r["model"] != "llama"), key=lambda r: r["scheduled_start_s"])
    for a, b in zip(order, order[1:]):
        gap = b["scheduled_start_s"] - a["end_s"]
        if gap > 5 and not a["decode_partners"]:
            out.append({"arm": arm, "kind": "dead_time_between_requests", "after": a["request_id"], "before": b["request_id"],
                        "gap_s": round(gap, 1)})
    return out


# ------------------------------------------------------------------------------------------------ io

def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def anchor_processes(logs: dict, requests: list[dict]) -> list[dict]:
    """Re-anchor each large-model server process to the dispatch time of its first request.

    parse_server_logs.py aligns processes through file mtimes, which carries a per-run offset of a
    few seconds (about +10 s in the baseline run). The launcher starts a model server at the dispatch
    of the first request that needs it, so that request's scheduled_start is the better anchor."""
    shifts = []
    for proc in logs["processes"]:
        if proc["role"] not in ("hot", "cold"):
            continue
        model = "qwen" if proc["role"] == "hot" else "gemma"
        window = (proc["start_s"] - 25, proc["start_s"] + (proc["loaded_after_s"] or 0) + 25)
        cands = [r for r in requests if r["model"] == model and window[0] <= r["scheduled_start_s"] <= window[1]]
        if not cands:
            continue
        shift = min(r["scheduled_start_s"] for r in cands) - proc["start_s"]
        shifts.append({"process": proc["index"], "shift_s": round(shift, 2)})
        proc["start_s"] += shift
        proc["end_s"] += shift
        for event in proc["dormant"]:
            event["t_s"] += shift
        for event in proc["controls"]:
            event["t_s"] += shift
        for pair in proc["mixed_skips"]:
            pair[0] += shift
        for entry in proc["timings"].values():
            for key in ("t_s", "released_t_s"):
                if key in entry:
                    entry[key] += shift
    return shifts


def switch_overhead(requests: list[dict], logs: dict) -> dict:
    """Time to first token of switch-leading requests minus their prompt time and the model load."""
    rows = []
    for r in requests:
        if r["model"] == "llama" or r["first_token_s"] is None:
            continue
        loads = [p for p in logs["processes"] if p["role"] in ("hot", "cold")
                 and r["scheduled_start_s"] - 1 <= p["start_s"] <= r["first_token_s"]]
        if not loads:
            continue
        extra = r["time_to_first_token_s"] - (r["server_prompt_ms"] or 0) / 1e3 - (loads[0]["loaded_after_s"] or 0)
        rows.append({"request_id": r["request_id"], "model": r["model"], "ttft_s": r["time_to_first_token_s"],
                     "load_s": loads[0]["loaded_after_s"], "prompt_s": round((r["server_prompt_ms"] or 0) / 1e3, 2),
                     "extra_s": round(extra, 2)})
    return {"switch_leading": rows,
            "mean_load_s": round(statistics.mean(r["load_s"] for r in rows), 1) if rows else None,
            "mean_extra_s": round(statistics.mean(r["extra_s"] for r in rows), 1) if rows else None,
            "total_switch_dead_time_s": round(sum(r["load_s"] + r["extra_s"] for r in rows), 1) if rows else None}


def analyze_arm(arm: str, run: pathlib.Path, logs_path: pathlib.Path, out_dir: pathlib.Path) -> dict:
    result = json.loads((run / "RESULT.json").read_text())
    logs = json.loads(logs_path.read_text())
    requests = request_rows(result, logs)
    alignment_shifts = anchor_processes(logs, requests)
    model_of = {row["request_id"]: model_key(row["model_id"]) for row in result["request_results"]}
    groups = longtail_groups(run)
    windows = window_rows(groups, model_of) if groups else []
    payload = {
        "arm": arm, "run": str(run), "status": result["status"], "duration_s": result["duration_us"] / 1e6,
        "host_kj": {k: v / 1e9 for k, v in result["trace_energy"]["fleet_energy_uj_by_domain"].items()},
        "requests": requests,
        "queue": queue_analysis(requests),
        "release_restore": release_restore(logs),
        "power": power_states(result, requests, windows, logs, run),
        "anomalies": anomalies(requests, logs, arm),
        "model_loads": [{"process": p["index"], "role": p["role"], "start_s": round(p["start_s"], 1), "load_s": p["loaded_after_s"]}
                        for p in logs["processes"]],
        "process_alignment_shifts": alignment_shifts,
        "switch_overhead": switch_overhead(requests, logs),
    }
    payload["switch_extra_s"] = payload["switch_overhead"]["mean_extra_s"]
    if windows:
        payload["token_attribution"] = token_attribution(result, windows, requests)
        payload["window_latency_energy"] = window_latency_energy(windows)
        payload["request_windows"] = request_window_summary(groups, windows)
        payload["phone_utilization"] = phone_utilization(result, requests, windows, logs)
        write_csv(out_dir / f"windows_{arm}.csv", windows)
        write_csv(out_dir / f"window_latency_energy_{arm}.csv", payload["window_latency_energy"])
        write_csv(out_dir / f"request_windows_{arm}.csv", payload["request_windows"])
    write_csv(out_dir / f"requests_{arm}.csv", requests)
    write_csv(out_dir / f"power_states_{arm}.csv", payload["power"]["states"])
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", type=pathlib.Path, required=True)
    parser.add_argument("--baseline", type=pathlib.Path, required=True)
    parser.add_argument("--treatment-logs", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-logs", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"treatment": analyze_arm("treatment", args.treatment, args.treatment_logs, args.out_dir),
               "baseline": analyze_arm("baseline", args.baseline, args.baseline_logs, args.out_dir)}
    (args.out_dir / "accounting.json").write_text(json.dumps(payload, indent=1) + "\n")
    t = payload["treatment"]
    print("token attribution (treatment):")
    for model, entry in t["token_attribution"].items():
        print(f"  {model:<6} tokens {entry['tokens']:>5} phone-policy {entry['phone_policy_tokens']:>5}"
              f" ({100 * (entry['phone_policy_share'] or 0):.0f}%) proof-assisted {entry['proof_assisted_tokens']:>5}")
        for category, tokens in entry["by_category"].items():
            print(f"      {category:<36} {tokens:>5}")
    print("power coarse states:")
    for arm in ("baseline", "treatment"):
        print(f"  {arm}: idle floor {payload[arm]['power']['idle_floor_w']} W")
        for state, entry in sorted(payload[arm]["power"]["coarse"].items()):
            print(f"      {state:<24} {entry['seconds']:>8.1f} s {entry['host_kj']:>8.2f} kJ {entry['host_w']!s:>6} W")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
