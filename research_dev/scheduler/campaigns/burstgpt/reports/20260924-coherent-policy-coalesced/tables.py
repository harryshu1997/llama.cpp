#!/usr/bin/env python3
"""Markdown tables from analyze_coherent_arm.py output (+ optional exactness JSONs)."""
import json
import sys


def main(path, *pairs):
    data = json.load(open(path))
    print("| arm | status | duration s | CPU kJ | GPU kJ | host kJ | host vs baseline | duration vs baseline | phone active s |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, arm in data.items():
        e = arm.get("energy") or {}
        fail = arm.get("failure")
        print(f"| {label} | {e.get('status') or ('FAILURE' if fail else '?')} | {e.get('duration_s')} | {e.get('cpu_kj')} | "
              f"{e.get('gpu_kj')} | {e.get('host_kj')} | {e.get('host_kj_vs_baseline_percent', '-')} % | "
              f"{e.get('duration_vs_baseline_percent', '-')} % | {e.get('phone_active_s')} |")
    print()
    print("| arm | model | output tokens | phone-policy window tokens | share | proof-assisted tokens | requests with phone calls |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for label, arm in data.items():
        for model, s in (arm.get("windows") or {}).get("phone_share", {}).items():
            print(f"| {label} | {model} | {s['output_tokens']} | {s['phone_policy_tokens']} | "
                  f"{s['phone_share_of_output']} | {s['proof_assisted_tokens']} | {s['requests_with_phone_calls']} |")
    print()
    print("| arm | server | processes | mixed passes | USB calls by rows | forward calls by rows | SHAPE calls by rows | multi-row USB calls |")
    print("| --- | --- | ---: | ---: | --- | --- | --- | ---: |")
    for label, arm in data.items():
        for role, s in (arm.get("server_logs") or {}).items():
            print(f"| {label} | {role} | {s['processes']} | {s['mixed_passes']} | {s['usb_calls_by_rows']} | "
                  f"{s['forward_calls_by_rows']} | {s['shape_calls_by_rows']} | {s['multi_row_usb_calls']} |")
    print()
    # A window's host energy covers every slot decoding in it; divided by active_batch it is per produced token.
    print("| arm | model | policy | active batch | tokens | windows | eligible | host J/slot-token | host J/produced token | ms/slot-token | phone calls |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, arm in data.items():
        for row in (arm.get("windows") or {}).get("by_model_policy_batch", []):
            j = row["host_j_per_token"]
            b = row["active_batch"] or 1
            print(f"| {label} | {row['model']} | {row['policy']} | {row['active_batch']} | {row['tokens']} | {row['windows']} | "
                  f"{row['measurement_eligible_windows']} | {j} | {None if j is None else round(j / b, 1)} | "
                  f"{row['ms_per_token']} | {row['phone_calls']} |")
    print()
    for label, arm in data.items():
        d = arm.get("decisions") or {}
        if d:
            print(f"- {label}: decisions {d.get('assistance_decision_reasons')}; coherence host decisions "
                  f"{d.get('coherence_decisions_fraction_zero')} by server reason "
                  f"{d.get('coherence_host_decisions_by_server_reason')}; coherence phone decisions "
                  f"{d.get('coherence_decisions_phone')}")
    print()
    print("| arm | request | model | windows | eligible (phone/host) | assumed-power windows | batches | host J/t | phone J/t | final ppm | host despite better phone | top decisions |")
    print("| --- | --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | --- | --- |")
    for label, arm in data.items():
        for r in arm.get("requests") or []:
            print(f"| {label} | {r['request']} | {r['model']} | {r['windows']} | {r['eligible_windows']} "
                  f"({r['eligible_phone_windows']}/{r['eligible_host_windows']}) | {r['assumed_power_windows']} | "
                  f"{r['active_batches']} | {r['host_j_per_token']} | {r['phone_j_per_token']} | {r['final_fraction_ppm']} | "
                  f"{'YES' if r['host_despite_better_phone'] else ''} | {'; '.join(r['top_decisions'])} |")
    peaks = {label: (arm.get("energy") or {}).get("active_slots_peak") for label, arm in data.items()}
    print(f"- active_slots_peak: {peaks}")
    for pair in pairs:
        label, _, p = pair.partition("=")
        r = json.load(open(p))
        print(f"- exactness {label}: {r['status']} {r['checks']}; identical {r['identical_outputs']}; "
              f"differences {r['output_differences']}; host saving {r['host_saving_percent']:.2f} %, "
              f"duration {r['duration_change_percent']:+.2f} %; by model {r['treatment_by_model']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
