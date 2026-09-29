#!/usr/bin/env python3
"""Per-device accounting of one run: which device sets the scheduler chose, when and why.

    python3 analyze_per_device.py RUN_DIR [--label NAME] [--out OUT.json]

RUN_DIR holds RESULT.json. Reports, per model:
  - phone calls and rows per proof session (OP15 HTP*, PIXEL*), i.e. per device;
  - measured windows per (device set, batch composition): tokens, fleet J/token, ms/token
    (LEARNING_WINDOW_RECORDED events; the device set is recorded only by per-device trees);
  - the server's device-set decisions over time (ASSISTANCE_DECISION `server_policy`: verdicts per
    batch composition, the pending proposal, dropped sets and their reasons);
  - decision records per (device set, batch composition, reason).
Older runs without device-set fields are reported by split fraction only.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = json.loads((args.run / "RESULT.json").read_text())
    label = args.label or args.run.name
    models = {row["request_id"]: row.get("model_id") for row in result.get("request_results", ())}

    calls = defaultdict(Counter)
    for proof in result.get("physical_execution_proofs", {}).values():
        ticket = proof.get("ticket_id", "")
        request_id = ticket.split(":attempt:")[0]
        for row in proof.get("phone_calls_by_session", ()):
            key = (models.get(request_id), row.get("session_id"))
            calls[key]["calls"] += int(row.get("calls", 0))
            calls[key]["rows"] += int(row.get("rows", 0))

    windows = defaultdict(lambda: {"tokens": 0, "energy_uj": 0, "latency_us": 0, "windows": 0})
    decisions = Counter()
    timeline = []
    last = {}
    for event in result.get("request_helper_events", ()):
        kind = event.get("kind")
        model = models.get(event.get("request_id"))
        if kind == "LEARNING_WINDOW_RECORDED":
            devices = "+".join(event["device_set"]) if "device_set" in event else "n/a"
            key = (model, devices, event.get("active_batch"), event.get("split_fraction_ppm"))
            row = windows[key]
            tokens = int(event.get("token_count", 0))
            row["windows"] += 1
            row["tokens"] += tokens
            row["energy_uj"] += int(event.get("fleet_energy_uj", 0))
            row["latency_us"] += int(event.get("latency_per_token_us", 0)) * max(1, tokens)
        elif kind == "ASSISTANCE_DECISION":
            devices = "+".join(event["device_set"]) if "device_set" in event else (
                "host" if event.get("selected_fraction_ppm") == 0 else "n/a")
            decisions[(model, devices or "host", event.get("active_batch"), event.get("reason"))] += 1
            server = event.get("server_policy") or {}
            state = {key: server.get(key) for key in (
                "verdict_device_sets", "proposal_device_set", "device_set_drops", "policy_device_set")}
            if any(value is not None for value in state.values()) and last.get(model) != state:
                last[model] = state
                timeline.append({"at_s": round(event["observed_at_us"] / 1e6, 1), "model": model,
                                 "request": event.get("request_id"), **state})

    report = {
        "label": label,
        "phone_calls_by_model_session": [
            {"model": model, "session": session, **dict(counter)}
            for (model, session), counter in sorted(calls.items(), key=lambda row: (str(row[0][0]), str(row[0][1])))],
        "windows_by_device_set_batch": [
            {"model": model, "device_set": devices, "active_batch": batch, "fraction_ppm": fraction,
             "windows": row["windows"], "tokens": row["tokens"] or None,
             # per slot token (fleet = host measured + assumed phones); None without token counts
             "fleet_J_per_token": (round(row["energy_uj"] / row["tokens"] / 1e6, 2) if row["tokens"] else None),
             "ms_per_token": (round(row["latency_us"] / row["tokens"] / 1e3, 1) if row["tokens"] else
                              round(row["latency_us"] / row["windows"] / 1e3, 1))}
            for (model, devices, batch, fraction), row in sorted(
                windows.items(), key=lambda row: tuple(str(value) for value in row[0]))],
        "decisions": [
            {"model": model, "device_set": devices, "active_batch": batch, "reason": reason, "count": count}
            for (model, devices, batch, reason), count in sorted(
                decisions.items(), key=lambda row: tuple(str(value) for value in row[0]))],
        "device_set_timeline": timeline,
    }
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print("==", label)
    for row in report["phone_calls_by_model_session"]:
        print("calls", row)
    for row in report["windows_by_device_set_batch"]:
        print("windows", row)
    for row in report["device_set_timeline"]:
        print("timeline", row)


if __name__ == "__main__":
    main()
