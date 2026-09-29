#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


USB_BYTES_PER_SECOND = 465_000_000.0
RPC_FLOOR_MS = 0.235
OUTPUT_JOIN_RESERVE_MS = 0.150
MIN_GAIN_PERCENT = 10.0
MIN_SAVING_MS = 0.100


def load_rows(path):
    rows = []
    with path.open() as source:
        for line in source:
            line = line.strip()
            if line.startswith("{"):
                rows.append(json.loads(line))
    return rows


def index_rows(rows):
    return {
        (row["k"], row["n"], row["m"], row["matrices"]): row
        for row in rows
    }


def transfer_ms(row):
    byte_count = (
        row["wire_input_f16_bytes"] + row["wire_output_f16_bytes"]
    )
    return byte_count / USB_BYTES_PER_SECOND * 1000.0


def phone_rpc_ms(row, percentile):
    return (
        row[f"compute_{percentile}_ms"]
        + RPC_FLOOR_MS
        + transfer_ms(row)
    )


def gain_percent(baseline, candidate):
    return (baseline - candidate) / baseline * 100.0


def qkv_at_m(cpu, phone, m, percentile):
    metric = f"compute_{percentile}_ms"
    baseline = cpu[(3840, 2048, m, 4)][metric]
    candidates = {
        "cpu": baseline,
        "q_half": max(
            cpu[(3840, 2048, m, 3)][metric],
            phone_rpc_ms(phone[(3840, 2048, m, 1)], percentile),
        ),
        "q": max(
            cpu[(3840, 2048, m, 2)][metric],
            phone_rpc_ms(phone[(3840, 4096, m, 1)], percentile),
        ),
        "kv": max(
            cpu[(3840, 4096, m, 1)][metric],
            phone_rpc_ms(phone[(3840, 2048, m, 2)], percentile),
        ),
    }
    choice = min(candidates, key=candidates.get)
    selected = candidates[choice]
    return {
        "baseline_ms": baseline,
        "candidates_ms": candidates,
        "choice": choice,
        "choice_ms": selected,
        "gain_percent": gain_percent(baseline, selected),
        "saving_ms": baseline - selected,
    }


def output_at_m(cpu, phone, m, percentile):
    metric = f"compute_{percentile}_ms"
    baseline = cpu[(4096, 3840, m, 1)][metric]
    raw_split = max(
        cpu[(2048, 3840, m, 1)][metric],
        phone_rpc_ms(phone[(2048, 3840, m, 1)], percentile),
    )
    reserved_split = raw_split + OUTPUT_JOIN_RESERVE_MS
    return {
        "baseline_ms": baseline,
        "raw_split_ms": raw_split,
        "reserved_split_ms": reserved_split,
        "raw_gain_percent": gain_percent(baseline, raw_split),
        "reserved_gain_percent": gain_percent(baseline, reserved_split),
        "reserved_saving_ms": baseline - reserved_split,
    }


def safe(values):
    return (
        values["gain_percent"] >= MIN_GAIN_PERCENT
        and values["saving_ms"] >= MIN_SAVING_MS
    )


def safe_output(values):
    return (
        values["reserved_gain_percent"] >= MIN_GAIN_PERCENT
        and values["reserved_saving_ms"] >= MIN_SAVING_MS
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cpu_results", type=Path)
    parser.add_argument("phone_results", type=Path)
    args = parser.parse_args()

    cpu = index_rows(load_rows(args.cpu_results))
    phone = index_rows(load_rows(args.phone_results))

    qkv_m = sorted({key[2] for key in cpu if key[:2] == (3840, 2048)})
    output_m = sorted({
        key[2] for key in cpu
        if key[0] == 4096 and key[1] == 3840 and key[3] == 1
    })

    qkv = []
    for m in qkv_m:
        p50 = qkv_at_m(cpu, phone, m, "p50")
        p90 = qkv_at_m(cpu, phone, m, "p90")
        qkv.append({
            "m": m,
            "p50": p50,
            "p90": p90,
            "safe": safe(p50) and safe(p90),
        })

    output_projection = []
    for m in output_m:
        p50 = output_at_m(cpu, phone, m, "p50")
        p90 = output_at_m(cpu, phone, m, "p90")
        output_projection.append({
            "m": m,
            "p50": p50,
            "p90": p90,
            "safe": safe_output(p50) and safe_output(p90),
        })

    result = {
        "assumptions": {
            "minimum_gain_percent": MIN_GAIN_PERCENT,
            "minimum_saving_ms": MIN_SAVING_MS,
            "output_join_reserve_ms": OUTPUT_JOIN_RESERVE_MS,
            "rpc_floor_ms": RPC_FLOOR_MS,
            "usb_MBps": USB_BYTES_PER_SECOND / 1e6,
        },
        "qkv": qkv,
        "output_projection": output_projection,
        "recommendations": {
            "qkv_safe_m": next(row["m"] for row in qkv if row["safe"]),
            "output_projection_safe_m": next(
                row["m"] for row in output_projection if row["safe"]
            ),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
