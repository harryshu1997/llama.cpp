#!/usr/bin/env python3
"""Phone re-provisioning timeline of a trace run (change #4), from RESULT.json + ADAPTIVE_DECODE_OBSERVATIONS.json.

    python3 analyze_reprovision.py --arm coherentEF=<run> --arm coherentRP=<run> --out X.json [--md X.md]

Per arm:
- layout: every READY layout (time, generation, phone layers and sessions per model, from the layout's shard layer
  masks), session loads (SESSION_LOADING -> SESSION_VERIFIED per session: model, bytes, seconds, MB/s), drains;
- desktop_reprovision blocks of EVALUATED events (mode, followed models, commitment source, leader, desktop load
  window, target/selected layers, fits_load_window, blocked/in-use sessions, learned rate), consecutive duplicates
  collapsed with a count;
- helper events: PREPARATION_DEFERRED by reason (WAITING_FOR_HELPER_RELEASE), PREPARATION_FAILED;
- per request: dispatch, first/last decode window, phone layers of its model resident at its first decode window
  and the maximum while it decoded, distinct layers its phone calls touched (physical_execution_proof);
- per model: phone layer-seconds of that model over the union of its decode windows (time-weighted mean layers).
Reads artifacts only.
"""
import argparse
import bisect
import collections
import json
import pathlib


def model_key(model_id):
    return model_id.split("-")[0]


def popcount(mask):
    return bin(int(mask)).count("1")


def layers_by_model(shards, model_of_artifact):
    layers, sessions = collections.Counter(), collections.Counter()
    for shard in shards or ():
        model = model_of_artifact.get(shard["artifact_sha256"], shard["artifact_sha256"][:15])
        layers[model] += popcount(shard["layer_mask"])
        sessions[model] += 1
    return dict(sorted(layers.items())), dict(sorted(sessions.items()))


def named(values, model_of_artifact):
    if isinstance(values, dict):
        return {model_of_artifact.get(k, k[:15]): v for k, v in sorted(values.items())}
    return [model_of_artifact.get(k, k[:15]) for k in values or ()]


def seconds(us):
    return None if us is None else round(us / 1e6, 1)


def union(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def analyze(run):
    result = json.loads((run / "RESULT.json").read_text())
    model_of_artifact = {row["artifact_sha256"]: model_key(model_id)
                         for model_id, row in result["model_artifacts"].items()}
    events = result.get("phone_residency_events") or []
    ready, loads, drains, open_loads = [], [], [], {}
    evaluated, collapsed = [], []
    kinds = collections.Counter(e.get("kind") for e in events)
    reasons = collections.Counter(e.get("reason") for e in events if e.get("kind") == "EVALUATED")
    for e in events:
        kind, at = e.get("kind"), e.get("observed_at_us")
        if kind == "READY":
            layers, sessions = layers_by_model(e["layout"]["shards"], model_of_artifact)
            ready.append({"at_s": seconds(at), "at_us": at, "generation": e.get("generation"),
                          "selection_reason": e.get("selection_reason"),
                          "changed_sessions": e["layout"].get("changed_session_ids"),
                          "layers": layers, "sessions": sessions})
        elif kind == "SESSION_LOADING":
            s = e["session"]
            open_loads[s["session_id"]] = (at, e.get("layout_generation"), s)
        elif kind == "SESSION_VERIFIED":
            s = e["session"]
            start = open_loads.pop(s["session_id"], None)
            if start is not None:
                duration = (at - start[0]) / 1e6
                loads.append({"session": s["session_id"], "layout_generation": start[1],
                              "model": model_of_artifact.get(s.get("resident_artifact_sha256"), "?"),
                              "bytes": s.get("resident_bytes"), "loading_s": seconds(start[0]),
                              "verified_s": seconds(at), "seconds": round(duration, 1),
                              "mb_per_s": round(s.get("resident_bytes", 0) / duration / 1e6, 1) if duration else None})
        elif kind == "SESSION_DRAINING":
            s = e.get("session") or {}
            drains.append({"at_s": seconds(at), "session": s.get("session_id"), "layout_generation": e.get("layout_generation"),
                           "model": model_of_artifact.get(s.get("resident_artifact_sha256"), "?")})
        elif kind == "EVALUATED" and e.get("desktop_reprovision"):
            d = e["desktop_reprovision"]
            row = {"at_s": seconds(at), "request": (e.get("request_id") or "").split(":")[-1],
                   "event_reason": e.get("reason"), "reason": d.get("reason"), "mode": d.get("mode"),
                   "followed": named(d.get("followed_artifact_sha256s"), model_of_artifact),
                   "split": named(d.get("split_artifact_sha256s"), model_of_artifact),
                   "source": d.get("desktop_commitment_source"),
                   "leader": (d.get("leader_request_id") or "").split(":")[-1] or None,
                   "load_window_s": [seconds(d.get("desktop_load_window_start_us")),
                                     seconds(d.get("desktop_load_window_end_us"))],
                   "target_layers": named(d.get("target_layers_by_artifact"), model_of_artifact),
                   "selected_layers": named(d.get("selected_layers_by_artifact"), model_of_artifact),
                   "target_sessions": named(d.get("target_session_counts_by_artifact"), model_of_artifact),
                   "fits_load_window": d.get("fits_load_window"),
                   "stage_swap_s": seconds(d.get("stage_swap_latency_us")),
                   "target_swap_s": seconds(d.get("target_swap_latency_us")),
                   "blocked": d.get("blocked_session_ids"), "in_use": d.get("in_use_session_ids"),
                   "rate_mb_s": round(d.get("load_bytes_per_second", 0) / 1e6, 1),
                   "learned_samples": d.get("learned_load_samples"),
                   "work": named(d.get("arrived_work_by_artifact"), model_of_artifact)}
            evaluated.append(row)
            key = {k: v for k, v in row.items() if k not in ("at_s", "request", "event_reason", "work")}
            if collapsed and collapsed[-1]["key"] == key:
                collapsed[-1]["count"] += 1
                collapsed[-1]["last_s"] = row["at_s"]
            else:
                collapsed.append({"key": key, "first_s": row["at_s"], "last_s": row["at_s"], "count": 1,
                                  "request": row["request"], "event_reason": row["event_reason"], "work": row["work"]})

    request_model = {row["request_id"]: model_key(row["model_id"]) for row in result["request_results"]}
    helper = collections.Counter()
    waiting = []
    for e in result.get("request_helper_events") or []:
        kind = e.get("kind")
        if kind in ("PREPARATION_DEFERRED", "PREPARATION_FAILED"):
            reason = e.get("reason") or (e.get("payload") or {}).get("reason")
            helper[(kind, reason)] += 1
            if reason == "WAITING_FOR_HELPER_RELEASE":
                waiting.append({"at_s": seconds(e.get("observed_at_us")),
                                "request": (e.get("request_id") or "").split(":")[-1],
                                "blocking": e.get("blocking_request_ids") or (e.get("payload") or {}).get("blocking_request_ids"),
                                "generation": e.get("phone_layout_generation")
                                or (e.get("payload") or {}).get("phone_layout_generation")})

    # layers per model over time: step function from READY events
    times = [row["at_us"] for row in ready]

    def layers_at(model, at_us):
        i = bisect.bisect_right(times, at_us) - 1
        return 0 if i < 0 else ready[i]["layers"].get(model, 0)

    windows = collections.defaultdict(list)
    obs_path = run / "ADAPTIVE_DECODE_OBSERVATIONS.json"
    if obs_path.exists():
        for group in json.loads(obs_path.read_text())["groups"]:
            rid = group["request_id"]
            for w in group["windows"]:
                windows[rid].append((w["started_at_us"], w["finished_at_us"], not w["policy"]["baseline"],
                                     w["token_end"] - w["token_start"]))
    requests = []
    for row in result["request_results"]:
        rid, model = row["request_id"], request_model[row["request_id"]]
        acquired = [r.get("observed_at_us") for r in row.get("dispatch_receipts") or [] if r.get("status") == "ACQUIRED"]
        proof = row.get("physical_execution_proof") or {}
        by_layer = [x for x in proof.get("phone_calls_by_layer") or [] if x.get("calls")]
        spans = windows.get(rid, [])
        entry = {"request": rid.split(":")[-1], "model": model, "output_tokens": row["output_tokens"],
                 "acquired_s": seconds(min(acquired)) if acquired else None,
                 "execution_started_s": seconds(((row.get("completion") or {}).get("execution_receipt") or {}).get("started_us")),
                 "end_s": seconds((row.get("completion") or {}).get("actual_end_us")),
                 "phone_calls": proof.get("phone_call_count", 0), "phone_layers_called": len(by_layer)}
        if spans:
            first, last = min(s for s, *_ in spans), max(e for _, e, *_ in spans)
            points = [first] + [t for t in times if first < t < last]
            entry.update({"first_window_s": seconds(first), "last_window_s": seconds(last),
                          "phone_layers_at_first_window": layers_at(model, first),
                          "max_phone_layers_while_decoding": max(layers_at(model, t) for t in points),
                          "phone_window_tokens": sum(t for *_, phone, t in spans if phone),
                          "window_tokens": sum(t for *_, t in spans)})
        requests.append(entry)

    per_model = {}
    by_model = collections.defaultdict(list)
    for rid, spans in windows.items():
        if rid in request_model:
            by_model[request_model[rid]].extend((s, e) for s, e, *_ in spans)
    for model, intervals in sorted(by_model.items()):
        total, weighted = 0, 0
        for a, b in union(intervals):
            points = [a] + [t for t in times if a < t < b] + [b]
            for x, y in zip(points, points[1:]):
                total += y - x
                weighted += layers_at(model, x) * (y - x)
        per_model[model] = {"decode_union_s": round(total / 1e6, 1),
                            "mean_phone_layers_while_decoding": round(weighted / total, 2) if total else None}
    return {"run": str(run), "event_kinds": dict(sorted(kinds.items())),
            "evaluated_reasons": dict(sorted((str(k), v) for k, v in reasons.items())),
            "ready_layouts": [{k: v for k, v in row.items() if k != "at_us"} for row in ready],
            "session_loads": loads, "session_drains": drains,
            "desktop_reprovision_events": len(evaluated),
            "desktop_reprovision_collapsed": [{**row["key"], "first_s": row["first_s"], "last_s": row["last_s"],
                                               "count": row["count"], "request": row["request"],
                                               "event_reason": row["event_reason"], "work": row["work"]}
                                              for row in collapsed],
            "helper_deferrals": {"|".join(map(str, k)): v for k, v in sorted(helper.items())},
            "waiting_for_helper_release": waiting,
            "requests": sorted(requests, key=lambda r: r["request"]),
            "per_model_decode_layers": per_model}


def markdown(report):
    lines = []
    for label, a in report.items():
        lines += [f"### {label}", "", f"- event kinds {a['event_kinds']}", f"- EVALUATED reasons {a['evaluated_reasons']}",
                  f"- helper deferrals {a['helper_deferrals']}; WAITING_FOR_HELPER_RELEASE {len(a['waiting_for_helper_release'])}",
                  f"- per model phone layers while decoding {a['per_model_decode_layers']}", "",
                  "| READY s | gen | changed | layers by model | sessions by model | selection |",
                  "| ---: | ---: | --- | --- | --- | --- |"]
        for r in a["ready_layouts"]:
            lines.append(f"| {r['at_s']} | {r['generation']} | {','.join(r['changed_sessions'] or [])} | "
                         f"{r['layers']} | {r['sessions']} | {r['selection_reason']} |")
        lines += ["", "| session | gen | model | GB | loading s | verified s | s | MB/s |", "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
        for r in a["session_loads"]:
            lines.append(f"| {r['session']} | {r['layout_generation']} | {r['model']} | {r['bytes'] / 1e9:.2f} | "
                         f"{r['loading_s']} | {r['verified_s']} | {r['seconds']} | {r['mb_per_s']} |")
        if a["desktop_reprovision_collapsed"]:
            lines += ["", "| s (count) | req | reason | mode | followed | source | leader | load window s | target layers | "
                      "selected layers | fits | stage/target swap s | blocked | rate MB/s (n) |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for r in a["desktop_reprovision_collapsed"]:
                span = f"{r['first_s']}" + (f"-{r['last_s']} ({r['count']})" if r["count"] > 1 else "")
                lines.append(f"| {span} | {r['request']} | {r['reason']} | {r['mode']} | {','.join(r['followed'])} | "
                             f"{r['source']} | {r['leader']} | {r['load_window_s']} | {r['target_layers']} | "
                             f"{r['selected_layers']} | {r['fits_load_window']} | {r['stage_swap_s']}/{r['target_swap_s']} | "
                             f"{r['blocked'] or '-'} | {r['rate_mb_s']} ({r['learned_samples']}) |")
        lines += ["", "| request | model | out | acquired s | exec start s | first/last window s | end s | "
                  "layers at first window | max layers while decoding | layers called | phone calls | phone/all window tokens |",
                  "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
        for r in a["requests"]:
            lines.append(f"| {r['request']} | {r['model']} | {r['output_tokens']} | {r['acquired_s']} | "
                         f"{r['execution_started_s']} | {r.get('first_window_s')}/{r.get('last_window_s')} | {r['end_s']} | "
                         f"{r.get('phone_layers_at_first_window', '-')} | {r.get('max_phone_layers_while_decoding', '-')} | "
                         f"{r['phone_layers_called']} | {r['phone_calls']} | "
                         f"{r.get('phone_window_tokens', '-')}/{r.get('window_tokens', '-')} |")
        if a["waiting_for_helper_release"]:
            lines += ["", "WAITING_FOR_HELPER_RELEASE: " + json.dumps(a["waiting_for_helper_release"])]
        lines.append("")
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
        report[label] = analyze(pathlib.Path(path))
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    if args.md:
        args.md.write_text(markdown(report) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
