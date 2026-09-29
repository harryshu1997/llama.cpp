"""Summarise decode-relocation KV gate arms: timing, host energy, memory footprint, KV headroom.

usage: analyze_gate.py ARM_DIR [ARM_DIR ...] [--budget-gib G] [--summary PATH]

The first directory is the reference (control) when two or more are given. Headroom is
``budget - peak footprint during decode`` expressed in CPU-tier KV tokens; it is a derived figure,
not a measured maximum context.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def rows_between(rows, start_ns, end_ns):
    return [row for row in rows if start_ns <= row["time_ns"] <= end_ns]


def stat(values, fn):
    values = [v for v in values if v is not None]
    return fn(values) if values else None


def gib(value):
    return None if value is None else round(value / 1024**3, 3)


def memory_window(rows, start_ns, end_ns):
    window = rows_between(rows, start_ns, end_ns)
    if not window:
        return None
    cur = [r["cgroup"].get("memory.current") for r in window if isinstance(r.get("cgroup"), dict)]
    cur = [c for c in cur if isinstance(c, int)]
    return {
        "samples": len(window),
        "vm_rss_min": stat([r["status"].get("VmRSS") for r in window], min),
        "vm_rss_max": stat([r["status"].get("VmRSS") for r in window], max),
        "vm_rss_end": window[-1]["status"].get("VmRSS"),
        "rss_anon_max": stat([r["status"].get("RssAnon") for r in window], max),
        "rss_file_min": stat([r["status"].get("RssFile") for r in window], min),
        "rss_file_max": stat([r["status"].get("RssFile") for r in window], max),
        "cgroup_current_min": min(cur) if cur else None,
        "cgroup_current_max": max(cur) if cur else None,
        "model_file_rss_min": stat([r.get("model_file_rss") for r in window], min),
        "model_file_rss_max": stat([r.get("model_file_rss") for r in window], max),
        "mem_available_min": stat([r["meminfo"].get("MemAvailable") for r in window], min),
    }


def progress_buckets(progress, first_ns, bucket=128):
    """ms per token per bucket of generated tokens, from the streaming progress callback."""
    points = sorted((p[0], p[1]) for p in progress if p[0] > 0)
    if len(points) < 2:
        return []
    out = []
    last_tokens, last_ns = 0, first_ns
    for tokens, t_ns in points:
        if tokens // bucket != last_tokens // bucket and tokens > last_tokens:
            out.append({"tokens": tokens, "ms_per_token": (t_ns - last_ns) / 1e6 / (tokens - last_tokens)})
            last_tokens, last_ns = tokens, t_ns
    tokens, t_ns = points[-1]
    if tokens > last_tokens:
        out.append({"tokens": tokens, "ms_per_token": (t_ns - last_ns) / 1e6 / (tokens - last_tokens)})
    return out


def energy_of(record, key):
    value = record.get(key) or {}
    cpu, gpu = value.get("cpu_package_energy_j"), value.get("gpu_board_energy_j")
    return {"cpu_j": cpu, "gpu_j": gpu, "host_j": None if cpu is None or gpu is None else cpu + gpu}


def summarise_arm(path, budgets):
    path = Path(path)
    result_path = path / "RESULT.json"
    record = load(result_path) if result_path.exists() else load(path / "FAILURE.json")
    plan = load(path / "KV_PLAN.json")
    rows = []
    memory_path = path / "MEMORY.jsonl"
    if memory_path.exists():
        for line in memory_path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    per_token = plan.get("cpu_kv_bytes_per_token")
    requests = []
    for req in record.get("requests", []):
        execution_path = path / f"EXECUTION-{req['request_id']}.json"
        execution = load(execution_path) if execution_path.exists() else {}
        first = req.get("first_token_ns")
        prefill_mem = memory_window(rows, req["started_ns"], first) if first else None
        decode_mem = memory_window(rows, first, req["finished_ns"]) if first else None
        entry = {
            "request_id": req["request_id"], "host_columns": req["host_columns"], "phone_columns": req["phone_columns"],
            "split": req["split"], "prefill_s": req.get("prefill_s"), "decode_s": req.get("decode_s"),
            "decode_ms_per_token": req.get("decode_ms_per_token"), "prompt_ms": req.get("prompt_ms"),
            "output_tokens": req.get("output_tokens"), "tokens_sha256": req.get("tokens_sha256"), "error": req.get("error"),
            "request_energy": energy_of(req, "request_host_energy"), "prefill_energy": energy_of(req, "prefill_host_energy"),
            "decode_energy": energy_of(req, "decode_host_energy"),
            "control_acks": [c.get("ack", {}).get("status") or c.get("ack") for c in req.get("controls", [])][:2],
            "phone_calls_mid_decode": ((req.get("phone_stats_mid_decode") or {}).get("runtime_stats") or req.get("phone_stats_mid_decode") or {}).get("calls")
                if isinstance(req.get("phone_stats_mid_decode"), dict) else None,
            "share_residency": {s.get("tag"): (s.get("share_residency") or {}).get("resident_fraction") for s in req.get("residency_samples", []) if s.get("share_residency")},
            "prefill_memory": prefill_mem, "decode_memory": decode_mem,
            "progress_buckets": progress_buckets(execution.get("progress", []), first) if first else [],
        }
        if decode_mem and per_token:
            # steady decode footprint = the last sample of the generation (KV only grows; the first-token
            # sample still shows the pre-release footprint)
            peak = decode_mem["vm_rss_end"]
            entry["headroom_tokens_by_budget_gib"] = {str(b): int((b * 1024**3 - peak) // per_token) for b in budgets} if peak else None
            if decode_mem.get("cgroup_current_max"):
                entry["cgroup_headroom_tokens_by_budget_gib"] = {str(b): int((b * 1024**3 - decode_mem["cgroup_current_max"]) // per_token) for b in budgets}
        requests.append(entry)
    proofs = record.get("dormant_proofs", [])
    return {
        "path": str(path), "arm": record.get("arm"), "status": record.get("status"), "dormant": record.get("dormant_host_share"),
        "kv_plan": {"cpu_layers": len(plan.get("cpu_layers", [])), "cpu_kv_bytes_per_token": per_token,
                    "bytes_by_pool": plan.get("bytes_by_pool")},
        "desktop_load_s": record.get("desktop_load_s"), "phone_preload_s": record.get("phone_preload_s"), "paid_s": record.get("paid_s"),
        "paid_energy": energy_of(record, "paid_host_energy"), "assumed_phone_j": record.get("assumed_phone_j"),
        "ready_rss": (record.get("memory_ready") or {}).get("vm_rss_bytes"),
        "finished_rss": (record.get("memory_finished") or {}).get("vm_rss_bytes"),
        "kv_lazy_lines": (record.get("memory_ready") or {}).get("kv_lazy_lines"),
        "dormant_proofs": proofs,
        "dormant_released_decode_bytes": [p["released_bytes"] for p in proofs if p.get("phase") == "decode"],
        "dormant_lower_bounds": record.get("dormant_lower_bounds"),
        "phone_proof_summary": record.get("phone_proof_summary"),
        "requests": requests,
    }


def fmt(value, digits=2):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render(arms):
    lines = []
    for arm in arms:
        lines.append(f"## {arm['arm']} ({arm['status']}) dormant={arm['dormant']}  {arm['path']}")
        lines.append(f"load {fmt(arm['desktop_load_s'])} s, phone preload {fmt(arm['phone_preload_s'])} s, paid {fmt(arm['paid_s'])} s, "
                     f"paid host energy CPU {fmt(arm['paid_energy']['cpu_j'], 0)} J + GPU {fmt(arm['paid_energy']['gpu_j'], 0)} J; "
                     f"RSS ready {gib(arm['ready_rss'])} GiB, finished {gib(arm['finished_rss'])} GiB; lazy KV lines {len(arm['kv_lazy_lines'] or [])}")
        if arm["dormant_released_decode_bytes"]:
            lines.append(f"dormant decode releases: {[gib(b) for b in arm['dormant_released_decode_bytes']]} GiB; lower bounds {arm['dormant_lower_bounds']}")
        lines.append("| request | host cols | phone cols | prefill s | decode ms/tok | req CPU J | req GPU J | decode CPU J | decode GPU J | RSS max prefill GiB | RSS min decode GiB | RSS end decode GiB | share resident (decode+20s) | phone calls | acks |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in arm["requests"]:
            pm, dm = r["prefill_memory"] or {}, r["decode_memory"] or {}
            lines.append(f"| {r['request_id']} | {r['host_columns']} | {r['phone_columns']} | {fmt(r['prefill_s'])} | {fmt(r['decode_ms_per_token'], 1)} | "
                         f"{fmt(r['request_energy']['cpu_j'], 0)} | {fmt(r['request_energy']['gpu_j'], 0)} | {fmt(r['decode_energy']['cpu_j'], 0)} | {fmt(r['decode_energy']['gpu_j'], 0)} | "
                         f"{fmt(gib(pm.get('vm_rss_max')), 3)} | {fmt(gib(dm.get('vm_rss_min')), 3)} | {fmt(gib(dm.get('vm_rss_end')), 3)} | "
                         f"{fmt(r['share_residency'].get('decode+20s'), 3)} | {fmt(r['phone_calls_mid_decode'])} | {r['control_acks']} |")
            if r.get("headroom_tokens_by_budget_gib"):
                lines.append(f"    headroom (RSS view) tokens by budget GiB: {r['headroom_tokens_by_budget_gib']}"
                             + (f"; cgroup view: {r['cgroup_headroom_tokens_by_budget_gib']}" if r.get("cgroup_headroom_tokens_by_budget_gib") else ""))
            if r["progress_buckets"]:
                lines.append("    ms/token by generated-token bucket: " + ", ".join(f"{b['tokens']}:{b['ms_per_token']:.0f}" for b in r["progress_buckets"]))
        lines.append("")
    if len(arms) >= 2 and all(a["requests"] for a in arms):
        ref, cmp = arms[0], arms[1]
        r0, r1 = ref["requests"][-1], cmp["requests"][-1]
        lines.append(f"## {cmp['arm']} vs {ref['arm']} (last request of each)")
        for key, a, b in (("prefill_s", r0["prefill_s"], r1["prefill_s"]), ("decode_ms_per_token", r0["decode_ms_per_token"], r1["decode_ms_per_token"]),
                          ("request host J", r0["request_energy"]["host_j"], r1["request_energy"]["host_j"]),
                          ("decode host J", r0["decode_energy"]["host_j"], r1["decode_energy"]["host_j"]),
                          ("paid host J", ref["paid_energy"]["host_j"], cmp["paid_energy"]["host_j"])):
            if a and b:
                lines.append(f"- {key}: {a:.2f} -> {b:.2f} ({(b - a) / a * 100:+.2f}%)")
        for power in ("3", "4.5", "6"):
            for mode in ("paid", "assisting"):
                try:
                    a = ref["paid_energy"]["host_j"] + ref["assumed_phone_j"][power][mode]
                    b = cmp["paid_energy"]["host_j"] + cmp["assumed_phone_j"][power][mode]
                    lines.append(f"- paid host + assumed phone {power} W ({mode}): {a:.0f} -> {b:.0f} J ({(b - a) / a * 100:+.2f}%)")
                except (KeyError, TypeError):
                    pass
        dm0, dm1 = r0["decode_memory"] or {}, r1["decode_memory"] or {}
        if dm0.get("vm_rss_end") and dm1.get("vm_rss_end"):
            lines.append(f"- end-of-decode RSS: {gib(dm0['vm_rss_end'])} -> {gib(dm1['vm_rss_end'])} GiB "
                         f"(freed {gib(dm0['vm_rss_end'] - dm1['vm_rss_end'])} GiB = {int((dm0['vm_rss_end'] - dm1['vm_rss_end']) // cmp['kv_plan']['cpu_kv_bytes_per_token'])} CPU-tier KV tokens at any fixed budget)")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", nargs="+")
    parser.add_argument("--budget-gib", type=float, nargs="*", default=[22.0, 24.0])
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    arms = [summarise_arm(path, args.budget_gib) for path in args.arms]
    print(render(arms))
    if args.summary:
        args.summary.write_text(json.dumps(arms, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
