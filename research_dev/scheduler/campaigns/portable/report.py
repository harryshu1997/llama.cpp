"""Reports of the portable path, from run directories only (no hardware).

    legs RUN                      per leg: ms per token, package / board / phone W, J per token
    sweep RUN [RUN ...]           shared legs against their neighbouring host-only legs, pooled per share,
                                  and the shared-layer fit of the last run (t_host, c_host, t_link, t_phone, f*)
    rows RUN [--skip N]           k requests decoding together at one share: ms and J per step and token
    seq RUN                       each request served twice back to back (alone, shared), per decoded token
    trace ALONE_RUN SHARED_RUN    two arms over one common window: CPU / GPU / phone kJ, latency

Energy = RAPL package + the serving GPU's board (``server_energy_summary``) + the phone's whole power
while it helps. The alone arm carries no phone energy. Every summary states the quality of the phone
energy: ``measured``, ``assumed`` (the site's stated power) or ``absent``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Mapping, Sequence

from ..._internal.layer_placement import LayerPlacementError
from ..._internal.shared_layer import bracketed_deltas, fit_shared_layer, pool_deltas
from ...drivers.meter import PhonePowerCalibration
from .rundir import IDLE, TAIL_IDLE, RunDirectory


def _pct(value: float | None) -> str:
    return "    n/a" if value is None else f"{value * 100:+6.1f}%"


def _watts(value: float | None) -> str:
    return "   n/a" if value is None else f"{value:6.2f}"


# --------------------------------------------------------------------------------------------------
# legs and sweep
# --------------------------------------------------------------------------------------------------


def legs_report(run: RunDirectory) -> dict[str, object]:
    """Every leg against the mean of the run's host-only legs."""
    legs = [dict(leg) for leg in run.legs()]
    host = [leg for leg in legs if leg["columns"] == 0]
    if not host:
        raise LayerPlacementError("a legs report needs at least one host-only leg")
    base_ms = statistics.fmean(leg["ms_token"] for leg in host)
    base_j = statistics.fmean(leg["total_j_token"] for leg in host)
    layers = run.layers_on_phone
    for leg in legs:
        leg["d_time"] = leg["ms_token"] / base_ms - 1.0
        leg["d_energy"] = leg["total_j_token"] / base_j - 1.0
        leg["ms_per_layer"] = (base_ms - leg["ms_token"]) / layers if layers else None
    hashes = {leg["tokens_sha256"] for leg in legs if leg["tokens_sha256"] is not None}
    identity = run.result.get("token_identity")
    if identity is None:
        identity = [len(hashes) <= 1]
    idle = run.idle(IDLE)
    return {"run": run.path.name, "idle": idle, "legs": legs, "token_identity": identity, "threads": run.threads,
            "helper": (run.config.get("helper") or {}).get("kind"), "layers_on_phone": layers,
            "phone_energy_quality": run.phone_quality()}


def format_legs(report: Mapping[str, object]) -> str:
    idle = report["idle"]
    lines = [f"{report['run']}: helper {report['helper']}, {report['layers_on_phone']} layers on the phone, "
             f"-t {report['threads']}, identity {report['token_identity']}, phone energy {report['phone_energy_quality']}",
             f"idle (server loaded): package {idle['cpu_w']:.1f} W, board {idle['gpu_w']:.1f} W, phone "
             + (f"{idle['phone']['w']:.2f} W" if idle["phone"] else "n/a"),
             "share   ms/token   d_time   pkg W   gpu W  phone W   host J/tok   total J/tok   d_energy   ms saved per phone layer"]
    for leg in report["legs"]:
        per_layer = "   n/a" if leg["ms_per_layer"] is None else f"{leg['ms_per_layer']:+6.2f}"
        lines.append(f"{leg['share'] * 100:5.1f}%  {leg['ms_token']:8.1f}  {_pct(leg['d_time'])}  {leg['cpu_w']:6.1f}  "
                     f"{leg['gpu_w']:6.1f}  {_watts(leg['phone_w'])}  {leg['host_j_token']:10.1f}  "
                     f"{leg['total_j_token']:11.1f}  {_pct(leg['d_energy'])}  {per_layer}")
    return "\n".join(lines)


def sweep_report(runs: Sequence[RunDirectory]) -> dict[str, object]:
    """Bracketed savings pooled over the runs, and the shared-layer fit of the last run that swept shares."""
    deltas = []
    for run in runs:
        for row in bracketed_deltas(run.legs()):
            deltas.append({**row, "run": run.path.name, "layers": run.layers_on_phone})
    pooled = pool_deltas(deltas)
    for share, row in pooled.items():
        row["runs"] = sorted({delta["run"] for delta in deltas if str(round(float(delta["share"]), 4)) == share})
    out: dict[str, object] = {"runs": [run.path.name for run in runs], "pooled": pooled, "legs": deltas,
                              "phone_energy_quality": sorted({run.phone_quality() for run in runs})}
    for run in reversed(runs):
        try:
            fit = fit_shared_layer(run.shape_rows(), n_ff=run.n_ff)
        except LayerPlacementError:
            continue
        by_share = {round(float(row["share"]), 4): row for row in deltas if row["run"] == run.path.name}
        rows = []
        for point in fit.points:
            delta = by_share.get(round(point.share, 4))
            rows.append({"share": point.share, "layer_ms": point.layer_ms, "model_ms": fit.layer_ms(point.share),
                         "host_ms": point.host_ms, "rpc_ms": point.rpc_ms, "compute_ms": point.compute_ms,
                         "d_time": delta["d_time"] if delta else None,
                         "d_energy": delta["d_energy"] if delta else None,
                         "dt_ms": delta["dt_ms"] if delta else None,
                         "dj": (delta["base_j"] - delta["j"]) if delta else None,
                         "phone_w": delta["phone_w"] if delta else None})
        out["sweep"] = {"run": run.path.name, "layers": run.layers_on_phone, "fit": fit.to_json(),
                        "f_plain": fit.f_plain, "t_plain_ms": (1.0 - fit.f_plain) * fit.t_host_ms, "rows": rows}
        break
    return out


def whole_layer_bound(report: Mapping[str, object], share: str) -> dict[str, float] | None:
    """Whole-layer offload against the bound P_F / (P_W + P_H), from a sweep that ran the full width.

    ``share`` names the pooled share whose marginal host power stands for ``P_F``. None when the sweep has
    no full-width leg or the share was not measured.
    """
    sweep = report.get("sweep")
    best = report["pooled"].get(share)
    if not sweep or not best or best.get("marginal_host_w") is None:
        return None
    whole = next((row for row in sweep["rows"] if row["share"] == 1.0 and row["dj"] is not None), None)
    if whole is None:
        return None
    layers, t_host = sweep["layers"], sweep["fit"]["t_host_ms"]
    p_f = best["marginal_host_w"]
    e_host = p_f * layers * t_host * 1e-3
    e_phone = e_host - whole["dj"]
    p_wait = e_phone / (layers * whole["layer_ms"] * 1e-3)
    return {"p_f_w": p_f, "p_w_plus_p_h_w": p_wait, "time_ratio": whole["layer_ms"] / t_host,
            "bound": p_f / p_wait, "energy_ratio": e_phone / e_host,
            "d_time_token": whole["d_time"], "d_energy_token": whole["d_energy"]}


def format_sweep(report: Mapping[str, object]) -> str:
    lines = ["pooled per share (each shared leg against its two neighbouring host-only legs); phone energy "
             + "/".join(report["phone_energy_quality"]),
             "share    n   time saving (min..max)     energy saving (min..max)    alone ms / J    shared ms / J   marginal host W"]
    for share, row in report["pooled"].items():
        marginal = "   n/a" if row["marginal_host_w"] is None else f"{row['marginal_host_w']:6.0f}"
        lines.append(f"{float(share) * 100:5.1f} % {row['n_legs']:3d}   {row['time_saving_mean']:5.1f} % "
                     f"({row['time_saving_min']:.1f}..{row['time_saving_max']:.1f})      "
                     f"{row['energy_saving_mean']:5.1f} % ({row['energy_saving_min']:.1f}..{row['energy_saving_max']:.1f})     "
                     f"{row['alone']['ms']:6.1f} / {row['alone']['total_j']:5.1f}   "
                     f"{row['shared']['ms']:6.1f} / {row['shared']['total_j']:5.1f}   {marginal}")
    sweep = report.get("sweep")
    if sweep:
        fit = sweep["fit"]
        lines += ["", f"fit of {sweep['run']} ({sweep['layers']} layers): host share = {fit['c_host_ms']:.2f} + (1-f) x "
                      f"{fit['t_host_ms']:.2f} ms, phone call = {fit['t_link_ms']:.2f} + f x {fit['t_phone_ms']:.2f} ms"
                      f"  ->  best share {fit['f_star'] * 100:.1f} %, layer {fit['t_star_ms']:.2f} ms "
                      f"({(fit['t_star_ms'] / fit['t_host_ms'] - 1) * 100:+.1f} %)",
                  "share    measured layer ms   model ms   host share ms   phone rpc ms (compute)   token time   token energy"]
        for row in sweep["rows"]:
            lines.append(f"{row['share'] * 100:5.1f} %  {row['layer_ms']:12.2f}   {row['model_ms']:9.2f}   "
                         f"{row['host_ms']:11.2f}   {row['rpc_ms']:9.2f} ({row['compute_ms']:.2f})          "
                         f"{_pct(row['d_time'])}      {_pct(row['d_energy'])}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------
# rows: k requests decoding together
# --------------------------------------------------------------------------------------------------


def rows_report(run: RunDirectory, skip: int = 32) -> dict[str, object]:
    out = []
    for phase in run.phases:
        requests = phase["requests"]

        def instant(row, microseconds):
            return row["first_token_ns"] + microseconds * 1000

        start = max(next(instant(row, us) for predicted, us in row["progress"] if predicted >= skip)
                    for row in requests)
        end = min(instant(row, row["progress"][-1][1]) for row in requests)
        steps = [sum(1 for _, us in row["progress"] if start < instant(row, us) <= end) for row in requests]
        energy = run.host(start, end)
        seconds = (end - start) / 1e9
        columns = int(requests[0]["columns"])
        phone = run.phone(start, end, active=columns > 0)
        total = energy["cpu_j"] + energy["gpu_j"] + (phone["w"] * seconds if phone and columns else 0.0)
        out.append({"phase": phase["name"], "k": len(requests), "columns": columns, "share": columns / run.n_ff,
                    "ms_step": seconds / statistics.fmean(steps) * 1e3, "cpu_w": energy["cpu_j"] / seconds,
                    "gpu_w": energy["gpu_j"] / seconds, "phone_w": phone["w"] if phone else None,
                    "j_step": total / statistics.fmean(steps), "j_token": total / sum(steps),
                    "tok_s": sum(steps) / seconds, "steps": steps})
    base = {row["k"]: row for row in out if row["columns"] == 0 and row["phase"] != "warm"}
    for row in out:
        alone = base.get(row["k"])
        row["d_time"] = row["ms_step"] / alone["ms_step"] - 1.0 if alone else None
        row["d_energy"] = row["j_token"] / alone["j_token"] - 1.0 if alone else None
    return {"run": run.path.name, "idle": run.idle(IDLE), "rows": out, "phone_energy_quality": run.phone_quality()}


def format_rows(report: Mapping[str, object]) -> str:
    lines = [f"{report['run']}: phone energy {report['phone_energy_quality']}",
             "phase       k  share  ms/step   d_time   pkg W   gpu W  phone W   J/step  J/token  d_energy   tok/s"]
    for row in report["rows"]:
        lines.append(f"{row['phase']:10s} {row['k']:2d} {row['share'] * 100:5.1f}% {row['ms_step']:8.1f}  "
                     f"{_pct(row['d_time'])}  {row['cpu_w']:6.1f}  {row['gpu_w']:6.1f}  {_watts(row['phone_w'])}"
                     f"  {row['j_step']:7.1f}  {row['j_token']:7.1f}   {_pct(row['d_energy'])}  {row['tok_s']:6.2f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------
# seq: the same request alone and shared, back to back
# --------------------------------------------------------------------------------------------------


def seq_report(run: RunDirectory) -> dict[str, object]:
    pairs: dict[object, dict[str, dict[str, object]]] = {}
    for phase in run.metered_phases():
        row = phase["requests"][0]
        first, end, submitted = int(row["first_token_ns"]), int(row["finished_ns"]), int(row["submitted_ns"])
        tokens = int(row["output_tokens"]) - 1
        decode, whole = run.host(first, end), run.host(submitted, end)
        shared = int(row["columns"]) > 0
        phone = run.phone(first, end) if shared else None
        phone_whole = run.phone(submitted, end) if shared else None
        entry = {"id": row["id"], "in": row["input_tokens"], "out": row["output_tokens"], "columns": row["columns"],
                 "decode_s": (end - first) / 1e9, "ms_token": (end - first) / 1e6 / tokens,
                 "cpu_j": decode["cpu_j"], "gpu_j": decode["gpu_j"],
                 "phone_j": phone["w"] * (end - first) / 1e9 if phone else 0.0,
                 "phone_w": phone["w"] if phone else None, "tokens": tokens,
                 "ttft_s": (first - submitted) / 1e9, "request_s": (end - submitted) / 1e9,
                 "request_j": whole["cpu_j"] + whole["gpu_j"]
                              + (phone_whole["w"] * (end - submitted) / 1e9 if phone_whole else 0.0),
                 "sha": row["tokens_sha256"]}
        entry["j"] = entry["cpu_j"] + entry["gpu_j"] + entry["phone_j"]
        entry["j_token"] = entry["j"] / tokens
        pairs.setdefault(row["id"], {})["shared" if shared else "alone"] = entry
    rows = []
    for request_id, pair in pairs.items():
        if len(pair) != 2:
            continue
        alone, shared = pair["alone"], pair["shared"]
        rows.append({"id": request_id, "alone": alone, "shared": shared,
                     "d_time": shared["ms_token"] / alone["ms_token"] - 1.0,
                     "d_energy": shared["j_token"] / alone["j_token"] - 1.0,
                     "d_request_s": shared["request_s"] / alone["request_s"] - 1.0,
                     "d_request_j": shared["request_j"] / alone["request_j"] - 1.0,
                     "same": alone["sha"] == shared["sha"]})
    if not rows:
        raise LayerPlacementError("a seq report needs requests served both alone and shared")

    def total(arm: str, key: str) -> float:
        return sum(row[arm][key] for row in rows)

    tokens = total("alone", "tokens")
    summary: dict[str, object] = {"requests": len(rows), "decode_tokens": tokens,
                                  "identical": sum(row["same"] for row in rows)}
    for arm in ("alone", "shared"):
        summary[arm] = {"ms_token": total(arm, "decode_s") * 1e3 / tokens, "cpu_j_token": total(arm, "cpu_j") / tokens,
                        "gpu_j_token": total(arm, "gpu_j") / tokens, "phone_j_token": total(arm, "phone_j") / tokens,
                        "j_token": total(arm, "j") / tokens, "request_s": total(arm, "request_s"),
                        "request_kj": total(arm, "request_j") / 1e3, "ttft_s_mean": total(arm, "ttft_s") / len(rows)}
    alone, shared = summary["alone"], summary["shared"]
    summary.update({
        "d_time": shared["ms_token"] / alone["ms_token"] - 1.0,
        "d_energy": shared["j_token"] / alone["j_token"] - 1.0,
        "d_time_range": [min(row["d_time"] for row in rows), max(row["d_time"] for row in rows)],
        "d_energy_range": [min(row["d_energy"] for row in rows), max(row["d_energy"] for row in rows)],
        "d_request_s": shared["request_s"] / alone["request_s"] - 1.0,
        "d_request_j": shared["request_kj"] / alone["request_kj"] - 1.0,
        "phone_energy_quality": run.phone_quality(),
    })
    return {"run": run.path.name, "summary": summary, "requests": rows}


def format_seq(report: Mapping[str, object]) -> str:
    summary = report["summary"]
    lines = [f"{report['run']}: phone energy {summary['phone_energy_quality']}",
             "request   in   out   ms/token alone  shared   d_time   J/token alone  shared  d_energy  phone W   "
             "request s alone/shared   request kJ alone/shared  same"]
    for row in report["requests"]:
        alone, shared = row["alone"], row["shared"]
        lines.append(f"{str(row['id'])[-3:]:>7s} {alone['in']:5d} {alone['out']:5d}   {alone['ms_token']:10.1f} "
                     f"{shared['ms_token']:8.1f}  {_pct(row['d_time'])}   {alone['j_token']:10.1f} "
                     f"{shared['j_token']:7.1f}  {_pct(row['d_energy'])}  {_watts(shared['phone_w'])}   "
                     f"{alone['request_s']:9.1f} {shared['request_s']:8.1f}   {alone['request_j'] / 1e3:9.2f} "
                     f"{shared['request_j'] / 1e3:8.2f}   {row['same']}")
    lines.append(f"pooled over {summary['requests']} requests, {summary['decode_tokens']} decode tokens "
                 f"({summary['identical']} identical outputs):")
    for arm in ("alone", "shared"):
        row = summary[arm]
        lines.append(f"  {arm:6s} {row['ms_token']:7.1f} ms/token  CPU {row['cpu_j_token']:6.1f}  GPU {row['gpu_j_token']:6.1f}  "
                     f"phone {row['phone_j_token']:5.2f}  total {row['j_token']:6.1f} J/token   whole requests "
                     f"{row['request_s']:7.1f} s, {row['request_kj']:7.1f} kJ, TTFT mean {row['ttft_s_mean']:.2f} s")
    lines.append(f"  decode: time {summary['d_time'] * 100:+.1f} % ({summary['d_time_range'][0] * 100:+.1f}.."
                 f"{summary['d_time_range'][1] * 100:+.1f}), energy {summary['d_energy'] * 100:+.1f} % "
                 f"({summary['d_energy_range'][0] * 100:+.1f}..{summary['d_energy_range'][1] * 100:+.1f}); whole requests: "
                 f"time {summary['d_request_s'] * 100:+.1f} %, energy {summary['d_request_j'] * 100:+.1f} %")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------
# trace: two arms over one fixed window
# --------------------------------------------------------------------------------------------------


def _arm(run: RunDirectory, window_s: float) -> dict[str, object]:
    phase = run.phases[-1]
    origin, end, requests = int(phase["origin_ns"]), int(phase["end_ns"]), phase["requests"]
    shared = run.result["mode"] == "shared"

    def energy(until_ns: int) -> dict[str, float]:
        measured_until = min(until_ns, run.last_sample_ns - 200_000_000)
        host = run.host(origin, measured_until)
        phone = run.phone(origin, measured_until) if shared else None
        out = {"cpu_kj": host["cpu_j"] / 1e3, "gpu_kj": host["gpu_j"] / 1e3,
               "phone_kj": phone["w"] * (measured_until - origin) / 1e12 if phone else 0.0, "extrapolated_s": 0.0}
        if until_ns > measured_until + 1_000_000_000:
            # the meters stopped before the common window closed: the tail is idle time at the measured idle power
            tail = run.idle(TAIL_IDLE)
            extra = (until_ns - measured_until) / 1e9
            out["cpu_kj"] += tail["cpu_w"] * extra / 1e3
            out["gpu_kj"] += tail["gpu_w"] * extra / 1e3
            if shared and tail["phone"]:
                out["phone_kj"] += tail["phone"]["w"] * extra / 1e3
            out["extrapolated_s"] = extra
        out["total_kj"] = out["cpu_kj"] + out["gpu_kj"] + out["phone_kj"]
        return out

    ttft = sorted((row["first_token_ns"] - row["target_ns"]) / 1e9 for row in requests)
    e2e = [(row["finished_ns"] - row["target_ns"]) / 1e9 for row in requests]
    tpot = [(row["finished_ns"] - row["first_token_ns"]) / 1e6 / (row["output_tokens"] - 1) for row in requests]
    events = sorted([(row["target_ns"], 1) for row in requests] + [(row["finished_ns"], -1) for row in requests])
    busy = active = peak = 0
    last = None
    for instant, change in events:
        if active > 0:
            busy += instant - last
        active += change
        last = instant
        peak = max(peak, active)
    decoding: dict[int, int] = {}
    for row in requests:       # decoded tokens by the number of requests decoding at that moment
        for _, microseconds in row["progress"]:
            instant = row["first_token_ns"] + microseconds * 1000
            count = sum(1 for other in requests if other["first_token_ns"] <= instant <= other["finished_ns"])
            decoding[count] = decoding.get(count, 0) + 1
    return {"mode": run.result["mode"], "run": run.path.name, "requests": len(requests),
            "output_tokens": sum(row["output_tokens"] for row in requests), "service_s": phase["service_s"],
            "busy_s": busy / 1e9, "peak_active": peak, "tokens_by_decoding_requests": dict(sorted(decoding.items())),
            "window": energy(origin + int(window_s * 1e9)), "service": energy(end),
            "e2e_mean_s": statistics.fmean(e2e), "e2e_p50_s": statistics.median(e2e), "e2e_max_s": max(e2e),
            "ttft_p50_s": statistics.median(ttft), "ttft_mean_s": statistics.fmean(ttft), "ttft_max_s": ttft[-1],
            "tpot_mean_ms": statistics.fmean(tpot), "tpot_p50_ms": statistics.median(tpot),
            "idle": run.idle(IDLE), "tail_idle": run.idle(TAIL_IDLE),
            "phone_service": run.phone(origin, end) if shared else None,
            "phone_calls": run.result.get("phone_calls"), "phone_energy_quality": run.phone_quality(),
            "per_request": [{"id": row["id"], "in": row["input_tokens"], "out": row["output_tokens"],
                             "ttft_s": (row["first_token_ns"] - row["target_ns"]) / 1e9,
                             "e2e_s": (row["finished_ns"] - row["target_ns"]) / 1e9,
                             "tpot_ms": (row["finished_ns"] - row["first_token_ns"]) / 1e6 / (row["output_tokens"] - 1),
                             "sha": row["tokens_sha256"]} for row in requests]}


def trace_report(alone_run: RunDirectory, shared_run: RunDirectory) -> dict[str, object]:
    window = max(alone_run.phases[-1]["service_s"], shared_run.phases[-1]["service_s"])
    alone, shared = _arm(alone_run, window), _arm(shared_run, window)
    same = [a["sha"] == b["sha"] for a, b in zip(alone["per_request"], shared["per_request"])]

    def change(key: str, select=lambda arm: arm) -> float:
        return select(shared)[key] / select(alone)[key] - 1.0

    return {"window_s": window, "alone": alone, "shared": shared, "identical_outputs": same,
            "saving_window_total": -change("total_kj", lambda arm: arm["window"]),
            "saving_window_cpu": -change("cpu_kj", lambda arm: arm["window"]),
            "saving_window_gpu": -change("gpu_kj", lambda arm: arm["window"]),
            "saving_service_total": -change("total_kj", lambda arm: arm["service"]),
            "d_service_s": change("service_s"), "d_busy_s": change("busy_s"), "d_e2e_mean": change("e2e_mean_s"),
            "d_ttft_p50": change("ttft_p50_s"), "d_tpot_mean": change("tpot_mean_ms")}


def format_trace(report: Mapping[str, object]) -> str:
    alone, shared, same = report["alone"], report["shared"], report["identical_outputs"]
    lines = [f"common window {report['window_s']:.1f} s; requests {alone['requests']}, output tokens "
             f"{alone['output_tokens']}; identical outputs {sum(same)}/{len(same)}; phone energy "
             f"{shared['phone_energy_quality']}",
             "arm      CPU kJ  GPU kJ  phone kJ  total kJ   service s  busy s  E2E mean  E2E p50  TTFT p50  TPOT mean  peak  extrap s"]
    for arm in (alone, shared):
        window = arm["window"]
        lines.append(f"{arm['mode']:7s} {window['cpu_kj']:7.1f} {window['gpu_kj']:7.1f} {window['phone_kj']:8.2f} "
                     f"{window['total_kj']:9.1f}  {arm['service_s']:9.1f} {arm['busy_s']:7.1f} {arm['e2e_mean_s']:9.1f} "
                     f"{arm['e2e_p50_s']:8.1f} {arm['ttft_p50_s']:9.2f} {arm['tpot_mean_ms']:10.1f} {arm['peak_active']:5d} "
                     f"{window['extrapolated_s']:8.1f}")
    lines.append(f"saving over the common window: total {report['saving_window_total'] * 100:.1f} % (CPU "
                 f"{report['saving_window_cpu'] * 100:.1f} %, GPU {report['saving_window_gpu'] * 100:.1f} %); over each "
                 f"arm's own service time {report['saving_service_total'] * 100:.1f} %")
    lines.append(f"time: service {report['d_service_s'] * 100:+.1f} %, busy {report['d_busy_s'] * 100:+.1f} %, E2E mean "
                 f"{report['d_e2e_mean'] * 100:+.1f} %, TTFT p50 {report['d_ttft_p50'] * 100:+.1f} %, TPOT mean "
                 f"{report['d_tpot_mean'] * 100:+.1f} %")
    lines.append("request   in   out   TTFT alone/shared s    E2E alone/shared s    TPOT alone/shared ms   same")
    for a, b, equal in zip(alone["per_request"], shared["per_request"], same):
        lines.append(f"{str(a['id'])[-3:]:>7s} {a['in']:5d} {a['out']:5d}   {a['ttft_s']:8.2f} {b['ttft_s']:8.2f}     "
                     f"{a['e2e_s']:8.1f} {b['e2e_s']:8.1f}     {a['tpot_ms']:8.1f} {b['tpot_ms']:8.1f}   {equal}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Report subcommands; shared by ``python3 -m ...portable report`` and this module's own ``main``."""
    parser.add_argument("--gauge-ratio", type=float,
                        help="phone power calibration for runs that recorded none (with --battery-unit / --battery-sign)")
    parser.add_argument("--battery-unit", choices=("mA", "uA"), default="mA")
    parser.add_argument("--battery-sign", type=int, choices=(1, -1), default=1)
    parser.add_argument("--json", type=Path, help="also write the report as JSON")
    sub = parser.add_subparsers(dest="report", required=True)
    sub.add_parser("legs").add_argument("run", type=Path)
    sub.add_parser("sweep").add_argument("run", type=Path, nargs="+")
    rows = sub.add_parser("rows")
    rows.add_argument("run", type=Path)
    rows.add_argument("--skip", type=int, default=32)
    sub.add_parser("seq").add_argument("run", type=Path)
    trace = sub.add_parser("trace")
    trace.add_argument("alone", type=Path)
    trace.add_argument("shared", type=Path)


def run_report(arguments: argparse.Namespace) -> dict[str, object]:
    calibration = None
    if arguments.gauge_ratio is not None:
        calibration = PhonePowerCalibration(arguments.battery_unit, arguments.battery_sign, arguments.gauge_ratio,
                                            "command line")

    def load(path: Path) -> RunDirectory:
        return RunDirectory(path, calibration)

    if arguments.report == "legs":
        report = legs_report(load(arguments.run))
        text = format_legs(report)
    elif arguments.report == "sweep":
        report = sweep_report([load(path) for path in arguments.run])
        text = format_sweep(report)
    elif arguments.report == "rows":
        report = rows_report(load(arguments.run), arguments.skip)
        text = format_rows(report)
    elif arguments.report == "seq":
        report = seq_report(load(arguments.run))
        text = format_seq(report)
    else:
        report = trace_report(load(arguments.alone), load(arguments.shared))
        text = format_trace(report)
    print(text)
    if arguments.json is not None:
        arguments.json.write_text(json.dumps(report, indent=1) + "\n", encoding="ascii")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    run_report(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
