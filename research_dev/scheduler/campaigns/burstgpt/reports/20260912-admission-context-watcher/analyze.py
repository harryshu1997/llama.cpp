"""Summarize one dev3 three-request run for the concurrency-context repair.

Reads run/RESULT.json + run/ADAPTIVE_DECODE_OBSERVATIONS.json and reports
per-window assistance, measurement-context changes, zero-assistance reasons,
coverage by layers/columns and fleet energy at assumed phone powers.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROLE_BY_SUFFIX = {"88120": "gemma", "88133": "qwen"}


def _role(request_id: str, result: dict) -> str:
    roles = result.get("model_roles") or {}
    for role, ids in roles.items():
        if isinstance(ids, (list, tuple)) and request_id in ids:
            return role
        if ids == request_id:
            return role
    return ROLE_BY_SUFFIX.get(request_id[-5:], request_id)


def window_table(group: dict) -> list[dict]:
    rows = []
    for w in group["windows"]:
        p = w["policy"]
        rows.append({
            "index": w["window_index"], "tokens": (w["token_start"], w["token_end"]),
            "start_s": round(w["started_at_us"] / 1e6, 3), "end_s": round(w["finished_at_us"] / 1e6, 3),
            "fraction_ppm": p["split_fraction_ppm"], "layers": len(p["layer_indices"]),
            "columns": p.get("columns", 0), "role": w.get("window_role"),
            "eligible": w.get("measurement_eligible", True), "batch": w["active_batch"],
            "ms_per_token": round(w["latency_per_token_us"] / 1e3, 1),
            "fleet_j_per_token": round(w["energy_per_token_uj"] / 1e6, 2),
            "phone_calls": w.get("completed_phone_calls", 0) or 0,
            "external": (w.get("external_activity_sha256") or "")[7:15] or None,
            "external_changed": w.get("external_activity_changed", False),
        })
    return rows


def coverage(group: dict, output_tokens: int) -> dict:
    assisted = 0
    weighted = 0.0
    by_mask = Counter()
    for w in group["windows"]:
        p = w["policy"]
        n = w["token_end"] - w["token_start"]
        if (w.get("completed_phone_calls") or 0) > 0 and p["split_fraction_ppm"] > 0:
            assisted += n
            weighted += n * p["split_fraction_ppm"] / 1e6
            by_mask[(len(p["layer_indices"]), p.get("columns", 0), p["split_fraction_ppm"])] += n
    return {
        "output_tokens": output_tokens,
        "assisted_tokens": assisted, "assisted_pct": round(100 * assisted / max(1, output_tokens), 2),
        "fraction_weighted_tokens": round(weighted, 2),
        "fraction_weighted_pct": round(100 * weighted / max(1, output_tokens), 2),
        "tokens_by_layers_columns_fraction": {f"L{k[0]}/C{k[1]}/{k[2]}ppm": v for k, v in sorted(by_mask.items())},
    }


def host_activity_summary(path: Path, bin_s: float = 5.0) -> dict | None:
    """Attribute host CPU power per bin: RAPL W, busy %, mean MHz, top processes."""
    rows = [json.loads(line) for line in open(path)] if path.exists() else []
    rows = [r for r in rows if r.get("host_activity") and r.get("rapl_package")]
    if len(rows) < 3:
        return None
    base = rows[0]["t_ns"]
    ticks_per_s = rows[0]["host_activity"]["clock_ticks_per_s"]
    bins = []
    comm_by_pid = {}
    rss_track = {}
    t = 0.0
    end = (rows[-1]["t_ns"] - base) / 1e9
    while t < end:
        sel = [r for r in rows if t <= (r["t_ns"] - base) / 1e9 < t + bin_s]
        if len(sel) >= 2:
            a, b = sel[0], sel[-1]
            dt = (b["rapl_package"]["sample_t_ns"] - a["rapl_package"]["sample_t_ns"]) / 1e9
            watts = (b["rapl_package"]["energy_uj"] - a["rapl_package"]["energy_uj"]) / 1e6 / dt if dt > 0 else None
            ja, jb = a["host_activity"]["cpu_jiffies"], b["host_activity"]["cpu_jiffies"]
            total = sum(jb[k] - ja[k] for k in jb)
            idle = (jb["idle"] - ja["idle"]) + (jb["iowait"] - ja["iowait"])
            busy_pct = None if total <= 0 else round(100.0 * (total - idle) / total, 1)
            khz = [r["host_activity"]["cpu_khz"]["mean"] for r in sel if r["host_activity"].get("cpu_khz")]
            # per-process CPU seconds in this bin from cumulative ticks (first/last seen)
            first_ticks, last_ticks = {}, {}
            for r in sel:
                for proc in r["host_activity"]["processes"]:
                    pid = proc["pid"]; comm_by_pid[pid] = proc["comm"]
                    first_ticks.setdefault(pid, proc["cpu_ticks"]); last_ticks[pid] = proc["cpu_ticks"]
                    if proc["rss_bytes"] >= 256 * 1024 * 1024:
                        rss_track.setdefault(pid, []).append((round((r["t_ns"] - base) / 1e9, 1), round(proc["rss_bytes"] / 1e9, 2)))
            seconds = {pid: (last_ticks[pid] - first_ticks[pid]) / ticks_per_s for pid in last_ticks}
            top = sorted(seconds.items(), key=lambda kv: -kv[1])[:4]
            # Runner threads: cumulative ticks per (pid, tid); creation time relative to
            # the owning process's own start so a thread can be tied to a request arrival.
            first_thread, last_thread, thread_meta = {}, {}, {}
            for r in sel:
                for th in r["host_activity"].get("threads", []):
                    key = (th["pid"], th["tid"])
                    first_thread.setdefault(key, th["cpu_ticks"]); last_thread[key] = th["cpu_ticks"]
                    thread_meta[key] = (th["comm"], th["start_ticks"])
            proc_start = {}
            for r in rows:
                for proc in r["host_activity"]["processes"]:
                    if "start_ticks" in proc:
                        proc_start.setdefault(proc["pid"], proc["start_ticks"])
            thread_seconds = {k: (last_thread[k] - first_thread[k]) / ticks_per_s for k in last_thread}
            top_threads = []
            for key, sec in sorted(thread_seconds.items(), key=lambda kv: -kv[1])[:3]:
                if sec <= 0:
                    continue
                comm, start = thread_meta[key]
                rel = None if key[0] not in proc_start else round((start - proc_start[key[0]]) / ticks_per_s, 1)
                top_threads.append(f"{comm}[{key[1]}]@+{rel}s:{sec:.2f}")
            bins.append({
                "t_s": round(t, 1), "rapl_w": None if watts is None else round(watts, 1), "cpu_busy_pct": busy_pct,
                "cores_busy": (None if busy_pct is None or not rows[0]["host_activity"].get("cpu_khz")
                               else round(busy_pct / 100 * rows[0]["host_activity"]["cpu_khz"]["count"], 2)),
                "mean_mhz": None if not khz else round(sum(khz) / len(khz) / 1000),
                "top_cpu_s": [f"{comm_by_pid[pid]}[{pid}]:{sec:.2f}" for pid, sec in top if sec > 0],
                "top_runner_threads_s": top_threads,
            })
        t += bin_s
    return {"bin_s": bin_s, "bins": bins,
            "large_rss_gb_by_pid": {f"{comm_by_pid.get(pid, '?')}[{pid}]": track[::max(1, len(track) // 12)]
                                    for pid, track in rss_track.items()}}


def main(run: Path) -> int:
    result = json.load(open(run / "RESULT.json"))
    store = json.load(open(run / "ADAPTIVE_DECODE_OBSERVATIONS.json"))
    events = result.get("request_helper_events", [])
    out = {"status": result.get("status"), "duration_s": round(result["duration_us"] / 1e6, 6),
           "counts": result.get("counts"), "helper_event_kinds": dict(Counter(e["kind"] for e in events)),
           "helper_event_requests": dict(Counter(e["request_id"] for e in events)),
           "helper_event_index_range": [min((e["event_index"] for e in events), default=None),
                                        max((e["event_index"] for e in events), default=None)],
           "requests": {}}
    lengths = {}
    for row in result.get("request_results", []):
        rid = row.get("request_id") or row.get("request", {}).get("request_id")
        lengths[rid] = row.get("output_tokens") or row.get("request", {}).get("output_tokens")
    for group in store["groups"]:
        rid = group["request_id"]
        role = _role(rid, result)
        req_events = [e for e in events if e["request_id"] == rid]
        table = window_table(group)
        changes = [e for e in req_events if e["kind"] == "CONTEXT_CHANGED"]
        decisions = [e for e in req_events if e["kind"] == "ASSISTANCE_DECISION"]
        zero = [{"token_index": e.get("token_index"), "reason": e.get("reason"), "at_us": e.get("observed_at_us"),
                 "incumbent_fraction_ppm": e.get("incumbent_fraction_ppm"), "external": (e.get("external_activity_sha256") or "")[7:15],
                 "remaining_probe_tokens": e.get("remaining_probe_tokens"), "eliminated": e.get("eliminated_policy_reasons")}
                for e in decisions if e.get("selected_fraction_ppm") == 0]
        # collapse consecutive identical zero reasons
        collapsed = []
        for z in zero:
            if collapsed and collapsed[-1]["reason"] == z["reason"] and collapsed[-1]["incumbent_fraction_ppm"] == z["incumbent_fraction_ppm"]:
                collapsed[-1]["until_token"] = z["token_index"]; collapsed[-1]["count"] += 1
            else:
                collapsed.append({**z, "until_token": z["token_index"], "count": 1})
        output_tokens = lengths.get(rid) or (table[-1]["tokens"][1] if table else 0)
        out["requests"][role] = {
            "request_id": rid, "terminal_status": group["terminal_status"], "state_history": group["state_history"],
            "final_fraction_ppm": group["final_policy"]["split_fraction_ppm"],
            "coverage": coverage(group, output_tokens),
            "external_contexts": sorted({r["external"] for r in table if r["external"]}),
            "external_changes": [{"window": r["index"], "tokens": r["tokens"], "end_s": r["end_s"], "to": r["external"]}
                                 for r in table if r["external_changed"]],
            "context_changed_events": [{k: e.get(k) for k in ("token_index", "reason", "external_desktop_ticket_ids",
                                        "previous_active_batch", "active_batch", "execution_compatible")} for e in changes],
            "zero_assistance_runs": collapsed,
            "assistance_decision_count": len(decisions),
            "windows": table,
        }
    trace = result.get("trace_energy") or {}
    meta = trace.get("estimation_metadata") or {}
    domains = trace.get("fleet_energy_uj_by_domain") or {}
    server_uj = sum(v for k, v in domains.items() if k != "phone-system")
    idle_uj = meta.get("phone_idle_power_mw", 0) * meta.get("phone_idle_time_ns", 0) / 1e6
    active_ns = meta.get("phone_active_time_ns", 0)
    out["fleet_energy_kj_by_assumed_phone_power"] = {
        f"{p}W": round((server_uj + idle_uj + p * 1000 * active_ns / 1e6) / 1e9, 6) for p in (3, 4.5, 6)}
    out["fleet_energy_uj_by_domain"] = domains
    out["phone_energy_evidence"] = meta.get("phone_energy_evidence")
    # Phase split per request: waiting for resources, launch/load/prefill to
    # first token, decode, and the run-level startup and cleanup tails.
    paid_start_ns = result["paid_start_ns"]
    phases = []
    for r in result.get("request_results", []):
        arrival_us = r.get("replay_arrival_us") or 0
        dispatch_us = min((x["observed_at_us"] for x in r.get("dispatch_receipts", [])
                           if x.get("status") == "ACQUIRED"), default=None)
        first_token_us = (None if not r.get("first_token_ns")
                          else (r["first_token_ns"] - paid_start_ns) / 1e3)
        end_us = (r.get("completion") or {}).get("actual_end_us")
        phases.append({
            "index": r.get("combined_request_index"),
            "wait_for_resources_s": None if dispatch_us is None else round((dispatch_us - arrival_us) / 1e6, 3),
            "launch_load_prefill_to_first_token_s": (
                None if dispatch_us is None or first_token_us is None
                else round((first_token_us - dispatch_us) / 1e6, 3)),
            "decode_s": None if first_token_us is None or end_us is None else round((end_us - first_token_us) / 1e6, 3),
            "terminal_proof_processing_s": "not separately timestamped in RESULT",
        })
    ends = [((r.get("completion") or {}).get("actual_end_us") or 0) for r in result.get("request_results", [])]
    first_dispatch = min((x["observed_at_us"] for r in result.get("request_results", [])
                          for x in r.get("dispatch_receipts", []) if x.get("status") == "ACQUIRED"), default=0)
    out["phase_split"] = {
        "startup_before_first_dispatch_s": round(first_dispatch / 1e6, 3),
        "cleanup_after_last_completion_s": round(result["duration_us"] / 1e6 - max(ends) / 1e6, 3),
        "requests": phases,
    }
    # Server-side startup split from llama-server logs: model file read
    # (load_tensors with mmap), warm-up/initialization, and listening.
    import glob, os, re
    stamp = re.compile(r"^(\d+)\.(\d\d)\.(\d\d\d)\.(\d\d\d) ")
    def seconds(line):
        m = stamp.match(line)
        return None if m is None else int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1e3 + int(m.group(4)) / 1e6
    servers = {}
    for path in sorted(glob.glob(str(run / "*.stderr"))):
        marks = {}
        with open(path, errors="replace") as handle:
            for line in handle:
                t = seconds(line)
                if t is None: continue
                if "load_model: loading model" in line and "load_start" not in marks: marks["load_start"] = t
                elif "load_tensors: loading model tensors" in line: marks["tensor_read_start"] = t
                elif "warming up the model" in line and "warmup_start" not in marks: marks["warmup_start"] = t
                elif "llama_server: listening on" in line and "listening" not in marks: marks["listening"] = t
        if marks:
            servers[os.path.basename(path)] = {
                "tensor_load_s": (None if "tensor_read_start" not in marks or "warmup_start" not in marks
                                  else round(marks["warmup_start"] - marks["tensor_read_start"], 3)),
                "tensor_load_note": ("load_tensors -> warm-up: file read plus any host-to-device transfer "
                                     "and buffer allocation; disk read is not isolated"),
                "warmup_and_init_s": (None if "warmup_start" not in marks or "listening" not in marks
                                      else round(marks["listening"] - marks["warmup_start"], 3)),
                "launch_to_listening_s": None if "listening" not in marks else round(marks["listening"], 3),
            }
    out["server_startup_split"] = servers
    out["host_activity"] = host_activity_summary(run / "resource-samples.jsonl")
    out["request_latency"] = [{
        "index": r.get("combined_request_index"), "executor": r.get("actual_executor_id"),
        "arrival_s": round((r.get("replay_arrival_us") or 0) / 1e6, 3), "recoveries": len(r.get("recoveries") or []),
        "execution_s": round((r.get("actual_latency_us") or 0) / 1e6, 6),
        "end_s": round(((r.get("completion") or {}).get("actual_end_us") or 0) / 1e6, 6),
        "output_tokens": lengths.get(r.get("request_id") or r.get("request", {}).get("request_id")),
    } for r in result.get("request_results", [])]
    json.dump(out, open(run.parent / "ANALYSIS.json", "w"), indent=1)
    for role, r in out["requests"].items():
        print(f"== {role} {r['request_id']} final={r['final_fraction_ppm']} status={r['terminal_status']}")
        print("   coverage", r["coverage"])
        print("   contexts", r["external_contexts"], "changes", r["external_changes"])
        print("   ctx events", r["context_changed_events"])
        print("   zero runs", r["zero_assistance_runs"])
    print("events", out["helper_event_kinds"], "index range", out["helper_event_index_range"])
    print("fleet kJ", out["fleet_energy_kj_by_assumed_phone_power"], "duration", out["duration_s"])
    print("latency", out["request_latency"])
    print("phases", out["phase_split"])
    print("servers", out["server_startup_split"])
    if out["host_activity"]:
        print("host activity bins (t, W, busy%, MHz, top):")
        for b in out["host_activity"]["bins"]:
            print("  ", b["t_s"], b["rapl_w"], b["cpu_busy_pct"], b["mean_mhz"], b["top_cpu_s"], b.get("top_runner_threads_s"))
        print("large RSS GB:", out["host_activity"]["large_rss_gb_by_pid"])
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
