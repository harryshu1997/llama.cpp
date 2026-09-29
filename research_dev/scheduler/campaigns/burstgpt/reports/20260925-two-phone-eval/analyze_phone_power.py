#!/usr/bin/env python3
"""Phone power diagnostic from the ~1 Hz sysfs samples run_chain_eval.py pulls next to a run (not a meter).

    python3 analyze_phone_power.py --run label=<dir holding NAME-POWER.json> [--name NAME] ... --out X.json

Per phone and sample: USB input power = usb/voltage_now x usb/current_now, battery power =
battery/voltage_now x battery/current_now (sign as reported by the driver), and their sum as the
phone-draw estimate under the convention that a positive battery current means discharge; the
opposite convention is also reported. Means are given over the whole run and, when the run has a
RESULT.json with execution receipts, over each model's execution window (first start to last end,
host-epoch mapped through the host/phone uptime anchors taken at sampler start and stop).
A second estimate adds the mean USB input to the battery discharge from the charge_counter slope
(coarse; the only battery term on the OP15, whose battery current_now reads 0). Sysfs currents are
instantaneous or firmware-averaged readings at ~1 Hz: a diagnostic, not a measurement of energy per token.
"""

import argparse
import json
from pathlib import Path
import statistics


def parse(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        row = {"uptime_s": float(fields[0])}
        for field in fields[1:]:
            key, _, value = field.partition("=")
            try:
                row[key] = int(value)
            except ValueError:
                pass
        rows.append(row)
    return rows


def watts(row, volts, amps):
    if volts not in row or amps not in row:
        return None
    return row[volts] * row[amps] * 1e-12


def summarize(rows):
    usb = [w for w in (watts(r, "usb/voltage_now", "usb/current_now") for r in rows) if w is not None]
    battery = [w for w in (watts(r, "battery/voltage_now", "battery/current_now") for r in rows) if w is not None]
    both = [(watts(r, "usb/voltage_now", "usb/current_now"), watts(r, "battery/voltage_now", "battery/current_now"))
            for r in rows]
    both = [(u, b) for u, b in both if u is not None and b is not None]
    mean = lambda values: round(statistics.fmean(values), 3) if values else None
    result = {"samples": len(rows), "usb_in_w_mean": mean(usb), "usb_in_w_max": round(max(usb), 3) if usb else None,
              "battery_w_mean_as_reported": mean(battery),
              "phone_w_if_positive_is_discharge": mean([u + b for u, b in both]),
              "phone_w_if_positive_is_charge": mean([u - b for u, b in both])}
    if rows and "usb/input_current_limit" in rows[0]:
        limit = rows[0]["usb/input_current_limit"]
        result["usb_input_limit_ma"] = limit // 1000
        result["samples_at_usb_limit"] = sum(1 for r in rows if r.get("usb/current_now", 0) >= 0.95 * limit)
    if len(rows) > 1 and "battery/charge_counter" in rows[0]:
        delta = rows[-1]["battery/charge_counter"] - rows[0]["battery/charge_counter"]
        seconds = rows[-1]["uptime_s"] - rows[0]["uptime_s"]
        volts = statistics.fmean(r["battery/voltage_now"] for r in rows) * 1e-6
        result["charge_counter_delta_uah"] = delta
        # coarse (counter resolution 1-2.5 mAh): battery discharge power from the counter slope
        result["battery_discharge_w_from_counter"] = round(-delta * 1e-6 * 3600 * volts / seconds, 3) if seconds > 0 else None
        if result["usb_in_w_mean"] is not None and result["battery_discharge_w_from_counter"] is not None:
            result["phone_w_usb_plus_counter"] = round(result["usb_in_w_mean"] + result["battery_discharge_w_from_counter"], 3)
    return result


def model_windows(run_dir, epoch_minus_monotonic_s):
    """Model execution windows in host epoch seconds (paid_start_ns is CLOCK_MONOTONIC)."""
    result_path = Path(run_dir) / "run-eval/run/RESULT.json"
    if not result_path.is_file() or epoch_minus_monotonic_s is None:
        return {}
    result = json.loads(result_path.read_text())
    started = result["paid_start_ns"] / 1e9 + epoch_minus_monotonic_s
    spans = {}
    for row in result["request_results"]:
        receipt = (row.get("completion") or {}).get("execution_receipt") or {}
        if receipt.get("started_us") and receipt.get("finished_us"):
            key = "qwen" if "qwen" in row["model_id"] else "gemma" if "gemma" in row["model_id"] else "other"
            first, last = spans.get(key, (receipt["started_us"], receipt["finished_us"]))
            spans[key] = (min(first, receipt["started_us"]), max(last, receipt["finished_us"]))
    return {key: (started + a / 1e6, started + b / 1e6) for key, (a, b) in spans.items()}


def analyze(directory, name):
    meta = json.loads((Path(directory) / (name + "-POWER.json")).read_text())
    anchor = next(iter(meta.values()))
    offset = (anchor["anchor_host_epoch_s"] - anchor["anchor_host_monotonic_s"]
              if "anchor_host_monotonic_s" in anchor else None)
    windows = model_windows(directory, offset)
    report = {}
    for phone, row in meta.items():
        rows = parse(row["local"] if Path(row["local"]).is_file() else Path(directory) / Path(row["local"]).name)
        span_phone = row["end_phone_uptime_s"] - row["anchor_phone_uptime_s"]
        span_host = row["end_host_epoch_s"] - row["anchor_host_epoch_s"]
        to_host = lambda uptime: row["anchor_host_epoch_s"] + (uptime - row["anchor_phone_uptime_s"])
        entry = {"run": summarize(rows), "clock_drift_s": round(span_host - span_phone, 3),
                 "duration_s": round(span_host, 1)}
        for model, (start, end) in windows.items():
            inside = [r for r in rows if start <= to_host(r["uptime_s"]) <= end]
            entry["window_" + model] = {"seconds": round(end - start, 1), **summarize(inside)}
        report[phone] = entry
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="label=dir[:NAME]")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = {}
    for spec in args.run:
        label, _, rest = spec.partition("=")
        directory, _, name = rest.partition(":")
        report[label] = analyze(directory, name or Path(directory).name + "-run")
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
