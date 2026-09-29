"""Summarize a many-request run (sparse_locality24) with renewal diagnostics.

Reads run/RESULT.json, run/ADAPTIVE_DECODE_OBSERVATIONS.json and
run/resource-samples.jsonl. Per request: role, executor, latency phases and
(for adaptive sessions) assistance coverage. Run-wide: LEASES_RENEWED
counts (coalesced renewal_count summed), HELPER_AUTHORIZATION_EXPIRED events,
renewal-thread CPU seconds from the per-thread host sampler, package power
bins and fleet energy at assumed phone powers.

usage: analyze_sparse24.py <run dir> [<output ANALYSIS.json>]
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import coverage, host_activity_summary, window_table  # noqa: E402


def role_of(model_id: str | None, result: dict) -> str:
    for role, mid in (result.get("model_roles") or {}).items():
        if mid == model_id:
            return role
    return model_id or "?"


def renewal_thread_seconds(path: Path) -> dict:
    """Cumulative CPU seconds of every runtime-lease-renewal thread and of the
    runner process, from the sampler's per-thread ticks (first vs last seen)."""
    if not path.exists():
        return {}
    first_t, last_t, meta = {}, {}, {}
    proc_first, proc_last, proc_comm = {}, {}, {}
    ticks_per_s = None
    base_ns = None
    for line in open(path):
        row = json.loads(line)
        if base_ns is None:
            base_ns = row["t_ns"]
        act = row.get("host_activity")
        if not act:
            continue
        ticks_per_s = act["clock_ticks_per_s"]
        for th in act.get("threads", []):
            key = (th["pid"], th["tid"])
            first_t.setdefault(key, (th["cpu_ticks"], row["t_ns"]))
            last_t[key] = (th["cpu_ticks"], row["t_ns"])
            meta[key] = th["comm"]
        for proc in act.get("processes", []):
            proc_first.setdefault(proc["pid"], proc["cpu_ticks"])
            proc_last[proc["pid"]] = proc["cpu_ticks"]
            proc_comm[proc["pid"]] = proc["comm"]
    if ticks_per_s is None:
        return {}
    renewal, watchers = {}, {}
    for key, comm in meta.items():
        sec = (last_t[key][0] - first_t[key][0]) / ticks_per_s
        row = {
            "cpu_s": round(sec, 3),
            "seen_s": [round((first_t[key][1] - base_ns) / 1e9, 1), round((last_t[key][1] - base_ns) / 1e9, 1)],
        }
        if comm.startswith("runtime-lease"):
            renewal[f"{comm}[{key[1]}]"] = row
        elif comm.startswith("request-helper"):
            watchers[f"{comm}[{key[1]}]"] = row
    top_procs = sorted(((proc_last[p] - proc_first[p]) / ticks_per_s, p) for p in proc_last)[-6:]
    return {
        "renewal_threads": renewal,
        "renewal_threads_total_cpu_s": round(sum(v["cpu_s"] for v in renewal.values()), 3),
        "request_helper_watcher_threads": watchers,
        "request_helper_watchers_total_cpu_s": round(sum(v["cpu_s"] for v in watchers.values()), 3),
        "top_process_cpu_s": [f"{proc_comm[p]}[{p}]:{sec:.1f}" for sec, p in reversed(top_procs)],
    }


def main(run: Path, output: Path | None) -> int:
    result = json.load(open(run / "RESULT.json"))
    store = json.load(open(run / "ADAPTIVE_DECODE_OBSERVATIONS.json"))
    events = result.get("request_helper_events", [])
    paid_start_ns = result["paid_start_ns"]
    duration_s = result["duration_us"] / 1e6
    out = {"status": result.get("status"), "duration_s": round(duration_s, 3), "counts": result.get("counts"),
           "helper_event_kinds": dict(Counter(e["kind"] for e in events)),
           "helper_event_index_range": [min((e["event_index"] for e in events), default=None),
                                        max((e["event_index"] for e in events), default=None)]}
    # Renewals and expiries per request.
    renewals = defaultdict(lambda: {"events": 0, "renewals": 0, "first_s": None, "last_s": None})
    expiries = []
    for e in events:
        if e["kind"] == "LEASES_RENEWED":
            r = renewals[e["request_id"]]
            r["events"] += 1
            r["renewals"] += int(e.get("renewal_count") or 1)
            t0 = e.get("first_observed_at_us", e.get("observed_at_us"))
            t1 = e.get("observed_at_us")
            if t0 is not None:
                r["first_s"] = round(t0 / 1e6, 3) if r["first_s"] is None else min(r["first_s"], round(t0 / 1e6, 3))
            if t1 is not None:
                r["last_s"] = round(t1 / 1e6, 3) if r["last_s"] is None else max(r["last_s"], round(t1 / 1e6, 3))
        elif e["kind"] == "HELPER_AUTHORIZATION_EXPIRED":
            expiries.append({"request_id": e["request_id"], "at_s": round((e.get("observed_at_us") or 0) / 1e6, 3),
                             "reason": e.get("reason"), "fraction_ppm": e.get("fraction_ppm"),
                             "lease_reserved_until_s": None if e.get("lease_reserved_until_us") is None
                             else round(e["lease_reserved_until_us"] / 1e6, 3)})
    total = sum(r["renewals"] for r in renewals.values())
    out["renewals"] = {"by_request": dict(renewals), "total": total,
                       "per_second_of_run": round(total / max(duration_s, 1e-9), 3),
                       "authorization_expiries": expiries}
    out["renewal_threads"] = renewal_thread_seconds(run / "resource-samples.jsonl")
    out["rejected_requests"] = result.get("rejected_requests", [])
    out["request_shape_preflight"] = {
        key: value for key, value in (result.get("request_shape_preflight") or {}).items() if key != "verdicts"
    }
    out["status_detail"] = {"status": result.get("status"), "counts": result.get("counts")}
    # Per request rows.
    groups = {g["request_id"]: g for g in store["groups"]}
    rows = []
    for r in result.get("request_results", []):
        rid = r["request_id"]
        arrival_us = r.get("replay_arrival_us") or 0
        dispatch_us = min((x["observed_at_us"] for x in r.get("dispatch_receipts", [])
                           if x.get("status") == "ACQUIRED"), default=None)
        first_token_us = None if not r.get("first_token_ns") else (r["first_token_ns"] - paid_start_ns) / 1e3
        end_us = (r.get("completion") or {}).get("actual_end_us")
        row = {"index": r.get("combined_request_index"), "role": role_of(r.get("model_id"), result), "request_id": rid,
               "executor": r.get("actual_executor_id"), "arrival_s": round(arrival_us / 1e6, 3),
               "wait_for_resources_s": None if dispatch_us is None else round((dispatch_us - arrival_us) / 1e6, 3),
               "to_first_token_s": (None if dispatch_us is None or first_token_us is None
                                    else round((first_token_us - dispatch_us) / 1e6, 3)),
               "decode_s": None if first_token_us is None or end_us is None else round((end_us - first_token_us) / 1e6, 3),
               "execution_s": round((r.get("actual_latency_us") or 0) / 1e6, 3),
               "end_s": None if end_us is None else round(end_us / 1e6, 3),
               "output_tokens": r.get("output_tokens"), "recoveries": len(r.get("recoveries") or []),
               "renewals": renewals[rid]["renewals"] if rid in renewals else 0}
        g = groups.get(rid)
        if g is not None:
            table = window_table(g)
            decisions = [e for e in events if e["request_id"] == rid and e["kind"] == "ASSISTANCE_DECISION"]
            zero = Counter(e.get("reason") for e in decisions if e.get("selected_fraction_ppm") == 0)
            row.update({"terminal_status": g["terminal_status"], "final_fraction_ppm": g["final_policy"]["split_fraction_ppm"],
                        "windows": len(table), "coverage": coverage(g, r.get("output_tokens") or 0),
                        "external_contexts": sorted({x["external"] for x in table if x["external"]}),
                        "zero_assistance_reasons": dict(zero),
                        "context_changed": sum(1 for e in events if e["request_id"] == rid and e["kind"] == "CONTEXT_CHANGED")})
        rows.append(row)
    rows.sort(key=lambda x: (x["arrival_s"], x["index"] or 0))
    out["requests"] = rows
    trace = result.get("trace_energy") or {}
    meta = trace.get("estimation_metadata") or {}
    domains = trace.get("fleet_energy_uj_by_domain") or {}
    server_uj = sum(v for k, v in domains.items() if k != "phone-system")
    idle_uj = meta.get("phone_idle_power_mw", 0) * meta.get("phone_idle_time_ns", 0) / 1e6
    active_ns = meta.get("phone_active_time_ns", 0)
    out["fleet_energy_kj_by_assumed_phone_power"] = {
        f"{p}W": round((server_uj + idle_uj + p * 1000 * active_ns / 1e6) / 1e9, 3) for p in (3, 4.5, 6)}
    out["fleet_energy_kj_by_domain"] = {k: round(v / 1e9, 3) for k, v in domains.items()}
    out["phone_energy_evidence"] = meta.get("phone_energy_evidence")
    host = host_activity_summary(run / "resource-samples.jsonl")
    out["host_activity"] = host
    if host:
        watts = [b["rapl_w"] for b in host["bins"] if b["rapl_w"] is not None and b["rapl_w"] >= 0]  # a negative bin is a counter wrap/reset
        renewal_bins = [b for b in host["bins"] if any(t.startswith("runtime-lease") for t in b.get("top_runner_threads_s", []))]
        out["host_summary"] = {
            "package_w_mean": round(sum(watts) / len(watts), 1) if watts else None,
            "package_w_max": max(watts) if watts else None,
            "bins_total": len(host["bins"]),
            "negative_power_bins_excluded": [(b["t_s"], b["rapl_w"]) for b in host["bins"] if b["rapl_w"] is not None and b["rapl_w"] < 0],
            "bins_with_renewal_thread_in_top3": len(renewal_bins),
            "renewal_thread_top_entries": [t for b in renewal_bins for t in b["top_runner_threads_s"] if t.startswith("runtime-lease")][:12],
        }
    json.dump(out, open(output or (run.parent / "ANALYSIS_SPARSE24.json"), "w"), indent=1)
    print("status", out["status"], "duration_s", out["duration_s"], "counts", out["counts"])
    print("renewals total", total, "per s", out["renewals"]["per_second_of_run"], "expiries", len(expiries))
    for rid, r in sorted(renewals.items(), key=lambda kv: -kv[1]["renewals"]):
        print("  ", rid, r)
    for x in expiries:
        print("   expiry", x)
    print("renewal threads total cpu s", out["renewal_threads"].get("renewal_threads_total_cpu_s"),
          "| request-helper watchers total cpu s", out["renewal_threads"].get("request_helper_watchers_total_cpu_s"),
          "n", len(out["renewal_threads"].get("request_helper_watcher_threads", {})))
    print("top procs", out["renewal_threads"].get("top_process_cpu_s"))
    print("rejected", out["rejected_requests"], "shape preflight", out["request_shape_preflight"])
    print("host", out.get("host_summary"))
    print("fleet kJ", out["fleet_energy_kj_by_assumed_phone_power"], out["fleet_energy_kj_by_domain"])
    print(f"{'idx':>4} {'role':<6} {'arr':>7} {'wait':>7} {'ttft':>7} {'decode':>7} {'end':>8} {'tok':>4} {'frac':>8} {'assist%':>7} {'ren':>5} executor")
    for r in rows:
        print(f"{r['index']:>4} {r['role']:<6} {r['arrival_s']:>7} {str(r['wait_for_resources_s']):>7} {str(r['to_first_token_s']):>7} "
              f"{str(r['decode_s']):>7} {str(r['end_s']):>8} {str(r['output_tokens']):>4} {str(r.get('final_fraction_ppm', '-')):>8} "
              f"{str((r.get('coverage') or {}).get('assisted_pct', '-')):>7} {r['renewals']:>5} {r['executor']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None))
