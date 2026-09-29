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
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
