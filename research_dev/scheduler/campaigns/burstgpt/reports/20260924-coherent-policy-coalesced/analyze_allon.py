#!/usr/bin/env python3
"""Accounting for the dev_v2 "all on" A/B under the new dispatcher (dev2baseDP, dev2allon vs dev2base, EF, RP, EF2).

    python3 analyze_allon.py --arm base=<run> --arm coherentEF=<run> ... --out X.json [--md X.md]

Superset of analyze_ef.py (energy, server counters, windows, decisions, markers, lock-out, intervals) plus:
- dispatch: RESULT `dispatch_policy` (policy, bypass counts, statistics), decision-log REPLAN reasons
  (`model_affinity_displaced`), displacement notes (`selected.dispatch_policy`), dispatch-receipt wake reasons,
  `placement_summary` (model_reload_count, transition_count);
- loads: every desktop llama-server launch (large-model-*.stderr) with its load seconds (`load_model` -> `model loaded`
  log timestamps), per-request load wait (execution start - ACQUIRED), model switch sequence;
- pairs: pairwise overlap of same-model execution intervals (execution receipts), decode-window overlap and tokens
  at active_batch >= 2 (from analyze_ef), RESULT active_slots_peak / active_server_slots maximum;
- probe: server-probe reasons (SERVER_PAIR_INCONCLUSIVE, SERVER_REFERENCE_BASELINE, SERVER_PAIR_NOT_IMPROVED,
  SERVER_PROBE_BUDGET_EXHAUSTED, CO_TENANT_POLICY_FOLLOW, ...) from the ASSISTANCE_DECISION events' server_policy
  snapshot and zero_assistance_reason, plus raw text counts.
Reads artifacts only.
"""
import argparse
import collections
import itertools
import json
import pathlib
import re

import analyze_coherent_arm as base
import analyze_ef as ef

LOAD_START = re.compile(r"^(\d+)\.(\d\d)\.(\d{3})\.(\d{3}) I srv\s+load_model: loading model", re.M)
LOAD_DONE = re.compile(r"^(\d+)\.(\d\d)\.(\d{3})\.(\d{3}) I srv\s+llama_server: model loaded", re.M)
PROBE_REASONS = ("SERVER_PAIR_INCONCLUSIVE", "SERVER_REFERENCE_BASELINE", "SERVER_PAIR_NOT_IMPROVED",
                 "SERVER_PROBE_BUDGET_EXHAUSTED", "SERVER_COMPARISON_HOST_WINDOW", "SERVER_PHONE_POLICY_FAILED",
                 "CO_TENANT_POLICY_FOLLOW", "SERVER_POLICY_COHERENCE", "model_affinity_displaced",
                 "MODEL_AFFINITY_DISPLACEMENT")


def stamp_seconds(match):
    minutes, seconds, ms, us = (int(x) for x in match.groups())
    return minutes * 60 + seconds + ms / 1e3 + us / 1e6


def loads(run, result):
    rows = []
    for path in sorted(run.glob("large-model-*-desktop*.stderr"), key=lambda p: int(p.name.split("-")[2])):
        role = "control" if "desktop-control" in path.name else ("hot" if "-hot-" in path.name else "cold")
        text = path.read_text(errors="replace")
        start, done = LOAD_START.search(text), LOAD_DONE.search(text)
        rows.append({"launch": int(path.name.split("-")[2]), "role": role,
                     "load_s": round(stamp_seconds(done) - stamp_seconds(start), 1) if start and done else None,
                     "loaded": bool(done)})
    requests = []
    sequence = []
    if result is not None:
        for row in sorted(result["request_results"], key=lambda r: r["replay_arrival_us"]):
            acquired = [r.get("observed_at_us") for r in row.get("dispatch_receipts") or [] if r.get("status") == "ACQUIRED"]
            receipt = (row.get("completion") or {}).get("execution_receipt") or {}
            started, finished = receipt.get("started_us"), receipt.get("finished_us")
            requests.append({"request": row["request_id"].split(":")[-1], "model": base.model_key(row["model_id"]),
                             "arrival_s": round(row["replay_arrival_us"] / 1e6, 1),
                             "acquired_s": round(min(acquired) / 1e6, 1) if acquired else None,
                             "exec_start_s": round(started / 1e6, 1) if started else None,
                             "exec_end_s": round(finished / 1e6, 1) if finished else None,
                             "load_wait_s": round((started - min(acquired)) / 1e6, 1) if acquired and started else None,
                             "wake_reasons": [r.get("wake_reason") for r in row.get("dispatch_receipts") or []],
                             "attempts": len(row.get("dispatch_receipts") or [])})
        for row in sorted(requests, key=lambda r: r["exec_start_s"] or 1e12):
            if row["model"] != "llama" and (not sequence or sequence[-1] != row["model"]):
                sequence.append(row["model"])
    summary = result.get("placement_summary") if result else None
    return {"server_launches": rows,
            "large_model_launches_by_role": dict(collections.Counter(r["role"] for r in rows)),
            "load_seconds_by_role": {role: [r["load_s"] for r in rows if r["role"] == role and r["load_s"] is not None]
                                     for role in ("hot", "cold", "control")},
            "total_load_s": round(sum(r["load_s"] or 0 for r in rows), 1),
            "placement_summary": None if summary is None else {k: summary.get(k) for k in ("model_reload_count", "transition_count")},
            "large_model_execution_sequence": sequence, "model_switches": max(0, len(sequence) - 1),
            "requests": requests}


def dispatch(run, result):
    out = {"result_dispatch_policy": None if result is None else result.get("dispatch_policy")}
    log_path = run / "SCHEDULER_DECISION_LOG.json"
    if log_path.exists():
        records = json.loads(log_path.read_text())["records"]
        kinds = collections.Counter(r.get("event_kind") for r in records)
        replans = collections.Counter(r.get("decision_reason") for r in records if r.get("event_kind") == "REPLAN")
        decisions = collections.Counter((r.get("event_kind"), r.get("decision_reason")) for r in records
                                        if r.get("event_kind") not in ("REPLAN",))
        notes = []
        for r in records:
            note = (r.get("selected") or {}).get("dispatch_policy")
            if note:
                notes.append({"at_s": round((r.get("event_time_us") or 0) / 1e6, 1), "event": r.get("event_kind"),
                              "requests": [x.split(":")[-1] for x in r.get("request_ids") or []],
                              "note": {k: (v if not isinstance(v, list) else [str(x).split(":")[-1] for x in v])
                                       for k, v in note.items()}})
        out.update({"decision_log_records": len(records), "event_kinds": dict(kinds),
                    "replan_reasons": dict(replans.most_common()),
                    "event_reasons": {"|".join(map(str, k)): v for k, v in decisions.most_common(40)},
                    "displacement_notes": notes})
    if result is not None:
        wake = collections.Counter(rec.get("wake_reason") for row in result["request_results"]
                                   for rec in row.get("dispatch_receipts") or [])
        out["dispatch_receipt_wake_reasons"] = dict(wake.most_common())
    return out


def pairs(result, detail):
    if result is None:
        return {}
    rows = []
    for row in result["request_results"]:
        receipt = (row.get("completion") or {}).get("execution_receipt") or {}
        if receipt.get("started_us") and receipt.get("finished_us"):
            rows.append((base.model_key(row["model_id"]), row["request_id"].split(":")[-1],
                         receipt["started_us"], receipt["finished_us"]))
    overlaps = []
    for (ma, a, sa, ea), (mb, b, sb, eb) in itertools.combinations(sorted(rows, key=lambda r: r[2]), 2):
        if ma != mb or ma == "llama":
            continue
        seconds = (min(ea, eb) - max(sa, sb)) / 1e6
        if seconds > 0:
            overlaps.append({"model": ma, "a": a, "b": b, "overlap_s": round(seconds, 1)})
    slots = result.get("active_server_slots") or []
    tokens = (detail or {}).get("window_tokens_by_model_policy_batch") or {}
    batch2 = {model: sum(t for policy in by.values() for b, t in policy.items() if int(b) >= 2)
              for model, by in tokens.items()}
    return {"execution_overlaps": overlaps, "execution_pairs": len(overlaps),
            "execution_overlap_s_by_model": {m: round(sum(o["overlap_s"] for o in overlaps if o["model"] == m), 1)
                                             for m in sorted({o["model"] for o in overlaps})},
            "decode_window_overlaps": (detail or {}).get("decode_span_overlaps"),
            "window_tokens_at_batch_ge2_by_model": batch2,
            "active_slots_peak": result.get("active_slots_peak"),
            "active_server_slots_max": max((s.get("active_slots", 0) for s in slots), default=None)}


def probe(run, result):
    reasons = collections.Counter()
    zero = collections.Counter()
    per_batch = collections.Counter()
    if result is not None:
        model_of = {row["request_id"]: base.model_key(row["model_id"]) for row in result["request_results"]}
        for event in result.get("request_helper_events") or []:
            if event.get("kind") != "ASSISTANCE_DECISION":
                continue
            model = model_of.get(event.get("request_id"), "?")
            server = event.get("server_policy") or {}
            if server.get("reason"):
                reasons[(model, str(server.get("reason")))] += 1
            for batch, reason in (server.get("reasons") or {}).items():
                per_batch[(model, str(batch), str(reason))] += 1
            for candidate in base._dicts(event):
                z = candidate.get("zero_assistance_reason")
                if z:
                    zero[(model, str(z))] += 1
    raw = {}
    for name in ("RESULT.json", "SCHEDULER_DECISION_LOG.json", "ADAPTIVE_DECODE_OBSERVATIONS.json"):
        path = run / name
        if path.exists():
            text = path.read_text(errors="replace")
            raw[name] = {k: text.count(k) for k in PROBE_REASONS if text.count(k)}
    return {"server_policy_reason_by_model": {"|".join(k): v for k, v in sorted(reasons.items())},
            "server_policy_reasons_per_batch_last_seen": {"|".join(k): v for k, v in sorted(per_batch.items())},
            "zero_assistance_reason_by_model": {"|".join(k): v for k, v in sorted(zero.items())},
            "raw_occurrences": raw}


def markdown(report):
    lines = ["## Dispatch, loads, pairs, probe reasons", ""]
    lines += ["| arm | dispatch_policy statistics | bypass counts | displacements (notes) | REPLAN reasons | wake reasons |",
              "| --- | --- | --- | --- | --- | --- |"]
    for label, arm in report.items():
        d = arm.get("dispatch") or {}
        rp = d.get("result_dispatch_policy") or {}
        lines.append(f"| {label} | {json.dumps(rp.get('statistics')) if rp else '-'} | "
                     f"{json.dumps(rp.get('bypass_counts')) if rp else '-'} | {len(d.get('displacement_notes') or [])} | "
                     f"{json.dumps(d.get('replan_reasons'))} | {json.dumps(d.get('dispatch_receipt_wake_reasons'))} |")
    lines += ["", "| arm | large-model launches (hot/cold/control) | load s by role | total load s | model_reload_count / transition_count | "
              "execution sequence | switches |", "| --- | --- | --- | ---: | --- | --- | ---: |"]
    for label, arm in report.items():
        l = arm.get("loads") or {}
        ps = l.get("placement_summary") or {}
        lines.append(f"| {label} | {json.dumps(l.get('large_model_launches_by_role'))} | {json.dumps(l.get('load_seconds_by_role'))} | "
                     f"{l.get('total_load_s')} | {ps.get('model_reload_count')} / {ps.get('transition_count')} | "
                     f"{' -> '.join(l.get('large_model_execution_sequence') or [])} | {l.get('model_switches')} |")
    lines += ["", "| arm | request | model | arrival s | acquired s | exec start s | exec end s | load wait s | attempts | wake reasons |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for label, arm in report.items():
        for r in (arm.get("loads") or {}).get("requests") or []:
            lines.append(f"| {label} | {r['request']} | {r['model']} | {r['arrival_s']} | {r['acquired_s']} | {r['exec_start_s']} | "
                         f"{r['exec_end_s']} | {r['load_wait_s']} | {r['attempts']} | {','.join(map(str, r['wake_reasons']))} |")
    lines += ["", "| arm | execution pairs (same model) | overlap s by model | decode-window overlaps | window tokens at batch>=2 | "
              "active_slots_peak / slots max |", "| --- | --- | --- | --- | --- | --- |"]
    for label, arm in report.items():
        p = arm.get("pairs") or {}
        ov = ", ".join(f"{o['model']} {o['a']}+{o['b']} {o['overlap_s']}" for o in p.get("execution_overlaps") or []) or "-"
        dw = ", ".join(f"{o['model']} {o['a']}+{o['b']} {o['overlap_s']}" for o in p.get("decode_window_overlaps") or []) or "-"
        lines.append(f"| {label} | {ov} | {json.dumps(p.get('execution_overlap_s_by_model'))} | {dw} | "
                     f"{json.dumps(p.get('window_tokens_at_batch_ge2_by_model'))} | {p.get('active_slots_peak')} / {p.get('active_server_slots_max')} |")
    lines += ["", "| arm | server_policy reason by model (decisions) | per-batch reasons (last seen) | zero_assistance_reason by model | raw |",
              "| --- | --- | --- | --- | --- |"]
    for label, arm in report.items():
        p = arm.get("probe") or {}
        lines.append(f"| {label} | {json.dumps(p.get('server_policy_reason_by_model'))} | "
                     f"{json.dumps(p.get('server_policy_reasons_per_batch_last_seen'))} | "
                     f"{json.dumps(p.get('zero_assistance_reason_by_model'))} | {json.dumps(p.get('raw_occurrences'))} |")
    lines += [""]
    for label, arm in report.items():
        for note in (arm.get("dispatch") or {}).get("displacement_notes") or []:
            lines.append(f"- {label} displacement at {note['at_s']} s ({note['event']}, requests {note['requests']}): {json.dumps(note['note'])}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="label=run dir")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--md", type=pathlib.Path)
    args = ap.parse_args()
    report = {}
    for spec in args.arm:
        label, _, path = spec.partition("=")
        run = pathlib.Path(path)
        result = base.load_result(run)
        detail = ef.window_detail(run, result)
        report[label] = {
            "run": str(run), "failure": (json.loads((run / "FAILURE.json").read_text())
                                         if (run / "FAILURE.json").exists() else None),
            "energy": base.energy(result) if result else None,
            "server_logs": base.server_logs(run),
            "windows": base.windows(run, result),
            "window_detail": detail,
            "decisions": base.decisions(run, result),
            "markers": ef.markers(run, result),
            "lockout": ef.lockout(result),
            "request_intervals": ef.request_intervals(result),
            "requests": base.per_request(run, result),
            "dispatch": dispatch(run, result),
            "loads": loads(run, result),
            "pairs": pairs(result, detail),
            "probe": probe(run, result),
        }
    labels = list(report)
    for label in labels[1:]:
        for ref in labels[:labels.index(label)]:
            e, b = report[label]["energy"], report[ref]["energy"]
            if e and b:
                e.setdefault("vs", {})[ref] = {
                    "host_kj_percent": round(100 * (e["host_kj"] / b["host_kj"] - 1), 2),
                    "duration_percent": round(100 * (e["duration_s"] / b["duration_s"] - 1), 2)}
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    if args.md:
        args.md.write_text(markdown(report) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
