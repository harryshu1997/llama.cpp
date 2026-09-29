#!/usr/bin/env python3
"""Per-arm accounting for the coherent-policy / coalesced-call change on the dev trace.

    python3 analyze_coherent_arm.py --arm baseline=<run dir> --arm plain=<run dir> --arm coherent=<run dir> --out X.json

Per arm: host kJ / duration (RESULT.json), decode tokens by (model, policy, active_batch) from the adaptive
windows, mixed forward passes (server counter "release skipped: mixed slot policies (N so far)"), phone call
row histogram (S41SERVERFFNUSB tokens=N per server role; S41SERVERFFNSHAPE calls with tokens>1), host J/token
and ms/token by (model, policy, active_batch), and the assistance decisions' zero-fraction reasons.
Everything is read from the artifacts; nothing runs on hardware.
"""
import argparse
import collections
import gzip
import json
import pathlib
import re

MIXED = re.compile(r"release skipped: mixed slot policies \((\d+) so far\)")
USB = re.compile(r"S41SERVERFFNUSB [^\n]*\btokens=(\d+)\b")
CALL = re.compile(r"S41SERVERFFNCALL [^\n]*\btokens=(\d+)\b")
SHAPE = re.compile(r"S41SERVERFFNSHAPE (\{.*\})")


def load_result(run):
    for name in ("RESULT.json", "RESULT.json.gz"):
        path = run / name
        if path.exists():
            with (gzip.open if name.endswith(".gz") else open)(path, "rt") as f:
                return json.load(f)
    return None


def model_key(model_id):
    for prefix in ("qwen", "gemma", "llama"):
        if model_id.startswith(prefix):
            return prefix
    return model_id


def energy(result):
    te = result.get("trace_energy") or {}
    dom = te.get("fleet_energy_uj_by_domain") or {}
    cpu, gpu, phone = (dom.get(k, 0) / 1e9 for k in ("cpu-package", "gpu-board", "phone-system"))
    duration = result.get("duration_us", 0) / 1e6
    return {"status": result.get("status"), "duration_s": round(duration, 1), "cpu_kj": round(cpu, 3),
            "gpu_kj": round(gpu, 3), "host_kj": round(cpu + gpu, 3), "phone_kj_assumed": round(phone, 3),
            "host_w": round((cpu + gpu) * 1e3 / duration, 1) if duration else None,
            "counts": result.get("counts"),
            "active_slots_peak": result.get("active_slots_peak"),
            "phone_active_s": round(((te.get("estimation_metadata") or {}).get("phone_active_time_ns") or 0) / 1e9, 1)}


def server_logs(run):
    roles = collections.defaultdict(lambda: {"mixed_passes": 0, "usb_rows": collections.Counter(),
                                             "call_rows": collections.Counter(), "shape_calls": collections.Counter(),
                                             "shape_rpc_ms": {}, "processes": 0})
    for path in sorted(run.glob("large-model-*.stderr")):
        role = "control" if "desktop-control" in path.name else ("hot" if "-hot-" in path.name else "cold")
        text = path.read_text(errors="replace")
        entry = roles[role]
        entry["processes"] += 1
        mixed = [int(m) for m in MIXED.findall(text)]
        entry["mixed_passes"] += max(mixed) if mixed else 0
        for rows in USB.findall(text):
            entry["usb_rows"][int(rows)] += 1
        for rows in CALL.findall(text):
            entry["call_rows"][int(rows)] += 1
        for raw in SHAPE.findall(text):
            shape = json.loads(raw)
            entry["shape_calls"][shape["tokens"]] += shape["calls"]
            entry["shape_rpc_ms"].setdefault(str(shape["tokens"]), []).append(round(shape["rpc_mean_ms"], 3))
    return {role: {"processes": e["processes"], "mixed_passes": e["mixed_passes"],
                   "usb_calls_by_rows": dict(sorted(e["usb_rows"].items())),
                   "forward_calls_by_rows": dict(sorted(e["call_rows"].items())),
                   "shape_calls_by_rows": dict(sorted(e["shape_calls"].items())),
                   "shape_rpc_mean_ms_by_rows": e["shape_rpc_ms"],
                   "multi_row_usb_calls": sum(c for r, c in e["usb_rows"].items() if r > 1)}
            for role, e in sorted(roles.items())}


def windows(run, result):
    store = run / "ADAPTIVE_DECODE_OBSERVATIONS.json"
    if not store.exists() or result is None:
        return {}
    model_of = {row["request_id"]: model_key(row["model_id"]) for row in result["request_results"]}
    outputs = collections.Counter()
    for row in result["request_results"]:
        outputs[model_key(row["model_id"])] += row["output_tokens"]
    by_key = collections.defaultdict(lambda: {"tokens": 0, "windows": 0, "eligible": 0, "host_uj": 0,
                                              "energy_tokens": 0, "duration_us": 0, "phone_calls": 0})
    phone_tokens = collections.Counter()
    window_tokens = collections.Counter()
    for group in json.loads(store.read_text())["groups"]:
        model = model_of.get(group["request_id"])
        if model is None:
            continue
        for w in group["windows"]:
            tokens = w["token_end"] - w["token_start"]
            policy = "phone" if not w["policy"]["baseline"] else "host"
            key = (model, policy, w.get("active_batch"))
            entry = by_key[key]
            entry["tokens"] += tokens
            entry["windows"] += 1
            window_tokens[model] += tokens
            if policy == "phone":
                phone_tokens[model] += tokens
            if not (w.get("output_valid") and w.get("failure_reason") is None):
                continue
            # Receipts omit measurement_eligible when it is true.
            entry["eligible"] += int(w.get("measurement_eligible", True) is not False)
            dom = w.get("fleet_energy_uj_by_domain") or {}
            entry["host_uj"] += dom.get("cpu-package", 0) + dom.get("gpu-board", 0)
            entry["energy_tokens"] += tokens
            entry["duration_us"] += w["finished_at_us"] - w["started_at_us"]
            entry["phone_calls"] += w.get("completed_phone_calls") or 0
    table = []
    for (model, policy, batch), e in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0)):
        table.append({"model": model, "policy": policy, "active_batch": batch, "tokens": e["tokens"],
                      "windows": e["windows"], "measurement_eligible_windows": e["eligible"],
                      "host_j_per_token": round(e["host_uj"] / e["energy_tokens"] / 1e6, 2) if e["energy_tokens"] else None,
                      "ms_per_token": round(e["duration_us"] / e["energy_tokens"] / 1e3, 1) if e["energy_tokens"] else None,
                      "phone_calls": e["phone_calls"]})
    assisted = collections.Counter()
    assisted_requests = collections.Counter()
    for row in result["request_results"]:
        by_layer = [entry["calls"] for entry in (row.get("physical_execution_proof") or {}).get("phone_calls_by_layer") or []
                    if isinstance(entry, dict) and "calls" in entry]
        if by_layer:
            # accounting rule: a request's assisted tokens are its phone calls per released layer
            assisted[model_key(row["model_id"])] += round(sum(by_layer) / len(by_layer))
            assisted_requests[model_key(row["model_id"])] += 1
    share = {model: {"output_tokens": outputs[model], "window_tokens": window_tokens[model],
                     "phone_policy_tokens": phone_tokens[model],
                     "phone_share_of_output": round(phone_tokens[model] / outputs[model], 3) if outputs[model] else None,
                     "proof_assisted_tokens": assisted[model], "requests_with_phone_calls": assisted_requests[model]}
             for model in sorted(outputs)}
    return {"by_model_policy_batch": table, "phone_share": share}


def per_request(run, result):
    """Per request: final policy, window eligibility and host J/token by policy (the 000/001 pattern check)."""
    store = run / "ADAPTIVE_DECODE_OBSERVATIONS.json"
    if not store.exists() or result is None:
        return []
    model_of = {row["request_id"]: model_key(row["model_id"]) for row in result["request_results"]}
    reasons = collections.defaultdict(collections.Counter)
    for event in result.get("request_helper_events") or []:
        if event.get("kind") == "ASSISTANCE_DECISION":
            reasons[event["request_id"]][(event.get("reason"), event.get("selected_fraction_ppm"))] += 1
    rows = []
    for group in json.loads(store.read_text())["groups"]:
        rid = group["request_id"]
        if rid not in model_of or not group["windows"]:
            continue
        ws = [w for w in group["windows"] if w.get("output_valid") and w.get("failure_reason") is None]

        def j_per_token(policy_is_host):
            chosen = [w for w in ws if w["policy"]["baseline"] == policy_is_host]
            tokens = sum(w["token_end"] - w["token_start"] for w in chosen)
            energy = sum((w.get("fleet_energy_uj_by_domain") or {}).get(d, 0)
                         for w in chosen for d in ("cpu-package", "gpu-board"))
            return round(energy / tokens / 1e6, 1) if tokens else None

        eligible = [w for w in ws if w.get("measurement_eligible", True) is not False]
        host_j, phone_j = j_per_token(True), j_per_token(False)
        final_host = group["final_policy"]["baseline"]
        rows.append({
            "request": rid.split(":")[-1], "model": model_of[rid], "windows": len(group["windows"]),
            "eligible_windows": len(eligible),
            "eligible_phone_windows": sum(not w["policy"]["baseline"] for w in eligible),
            "eligible_host_windows": sum(w["policy"]["baseline"] for w in eligible),
            "assumed_power_windows": sum("ASSUMED_4P5W" in (w.get("evidence_ids") or []) for w in ws),
            "energy_kinds": sorted({w.get("energy_attribution_kind") for w in ws}),
            "active_batches": sorted({w.get("active_batch") for w in ws}),
            "host_j_per_token": host_j, "phone_j_per_token": phone_j,
            "final_fraction_ppm": group["final_policy"]["split_fraction_ppm"],
            "host_despite_better_phone": bool(final_host and host_j and phone_j and phone_j < host_j),
            "top_decisions": ["%s@%s x%d" % (r, f, n) for (r, f), n in reasons[rid].most_common(4)],
        })
    return sorted(rows, key=lambda row: row["request"])


def decisions(run, result):
    """Assistance decisions come from the request helper events recorded in RESULT.json."""
    if result is None:
        return {}
    reasons = collections.Counter()
    coherent_zero = 0
    coherent_phone = 0
    by_model_reason = collections.Counter()
    server_host_reasons = collections.Counter()
    model_of = {row["request_id"]: model_key(row["model_id"]) for row in result["request_results"]}
    for event in result.get("request_helper_events") or []:
        if event.get("kind") != "ASSISTANCE_DECISION":
            continue
        for candidate in _dicts(event):
            if candidate.get("reason") is None or "selected_fraction_ppm" not in candidate:
                continue
            reasons[candidate["reason"]] += 1
            by_model_reason[(model_of.get(event.get("request_id"), "?"), candidate["reason"],
                             "phone" if candidate["selected_fraction_ppm"] else "host")] += 1
            if candidate["reason"] == "SERVER_POLICY_COHERENCE":
                if candidate["selected_fraction_ppm"] == 0:
                    coherent_zero += 1
                    server = event.get("server_policy") or {}
                    server_host_reasons[str(server.get("reason"))] += 1
                else:
                    coherent_phone += 1
            break
    return {"assistance_decision_reasons": dict(reasons.most_common()),
            "by_model_reason_selected": {"|".join(k): v for k, v in sorted(by_model_reason.items())},
            "coherence_decisions_fraction_zero": coherent_zero, "coherence_decisions_phone": coherent_phone,
            "coherence_host_decisions_by_server_reason": dict(server_host_reasons.most_common()),
            "final_server_policy": _final_server_policy(result)}


def _final_server_policy(result):
    """Last recorded shared decision per request (verdicts by batch, attempts, reason)."""
    last = {}
    for event in result.get("request_helper_events") or []:
        if event.get("kind") == "ASSISTANCE_DECISION" and event.get("server_policy"):
            last[event["request_id"].split(":")[-1]] = event["server_policy"]
    return last


def _dicts(value):
    if isinstance(value, dict):
        yield value
        for v in value.values():
            yield from _dicts(v)
    elif isinstance(value, list):
        for v in value:
            yield from _dicts(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", action="append", required=True, help="label=run dir")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()
    report = {}
    for spec in args.arm:
        label, _, path = spec.partition("=")
        run = pathlib.Path(path)
        result = load_result(run)
        report[label] = {
            "run": str(run), "failure": (json.loads((run / "FAILURE.json").read_text())
                                         if (run / "FAILURE.json").exists() else None),
            "energy": energy(result) if result else None,
            "server_logs": server_logs(run),
            "windows": windows(run, result),
            "decisions": decisions(run, result),
            "requests": per_request(run, result),
        }
    if len(report) > 1:
        labels = list(report)
        base = report[labels[0]]["energy"]
        for label in labels[1:]:
            e = report[label]["energy"]
            if base and e:
                e["host_kj_vs_" + labels[0] + "_percent"] = round(100 * (e["host_kj"] / base["host_kj"] - 1), 2)
                e["duration_vs_" + labels[0] + "_percent"] = round(100 * (e["duration_s"] / base["duration_s"] - 1), 2)
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
