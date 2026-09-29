#!/usr/bin/env python3
"""Markdown tables from analyze_ef.py output: tables_ef.py ANALYSIS.json [label=PAIR.json ...]."""
import json
import sys


def fmt(value):
    return "-" if value is None else value


def main(path, *pairs):
    data = json.load(open(path))
    print("| arm | status | duration s | CPU kJ | GPU kJ | host kJ | host W | host kJ / duration vs earlier arms |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for label, arm in data.items():
        e = arm.get("energy") or {}
        vs = "; ".join(f"{ref} {v['host_kj_percent']:+.1f} % / {v['duration_percent']:+.1f} %"
                       for ref, v in (e.get("vs") or {}).items()) or "-"
        print(f"| {label} | {e.get('status') or ('FAILURE' if arm.get('failure') else '?')} | {e.get('duration_s')} | "
              f"{e.get('cpu_kj')} | {e.get('gpu_kj')} | {e.get('host_kj')} | {e.get('host_w')} | {vs} |")
    print()
    print("| arm | model | output tokens | phone-policy window tokens | share | requests with phone calls |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for label, arm in data.items():
        for model, s in (arm.get("windows") or {}).get("phone_share", {}).items():
            if model == "llama":
                continue
            print(f"| {label} | {model} | {s['output_tokens']} | {s['phone_policy_tokens']} | "
                  f"{s['phone_share_of_output']} | {s['requests_with_phone_calls']} |")
    print()
    print("| arm | server | mixed passes | SHAPE calls by rows | USB calls by rows | multi-row USB calls |")
    print("| --- | --- | ---: | --- | --- | ---: |")
    for label, arm in data.items():
        for role, s in (arm.get("server_logs") or {}).items():
            if role == "control":
                continue
            print(f"| {label} | {role} | {s['mixed_passes']} | {s['shape_calls_by_rows']} | {s['usb_calls_by_rows']} | "
                  f"{s['multi_row_usb_calls']} |")
    print()
    print("| arm | model | window tokens by policy and active batch | same-model decode overlaps (s) |")
    print("| --- | --- | --- | --- |")
    for label, arm in data.items():
        d = arm.get("window_detail") or {}
        for model, t in (d.get("window_tokens_by_model_policy_batch") or {}).items():
            overlaps = [f"{o['a']}+{o['b']} {o['overlap_s']}" for o in d.get("decode_span_overlaps") or []
                        if o["model"] == model]
            print(f"| {label} | {model} | {json.dumps(t, sort_keys=True)} | {', '.join(overlaps) or '-'} |")
    print()
    print("| arm | model | policy | batch | scope | windows | tokens | fleet J/slot-token | fleet J/produced token | "
          "host J/produced token | ms/slot-token | rows/call |")
    print("| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, arm in data.items():
        for r in (arm.get("window_detail") or {}).get("by_model_policy_batch_scope", []):
            print(f"| {label} | {r['model']} | {r['policy']} | {r['active_batch']} | {r['scope']} | {r['windows']} | "
                  f"{r['tokens']} | {r['fleet_j_per_slot_token']} | {r['fleet_j_per_produced_token']} | "
                  f"{r['host_j_per_produced_token']} | {r['ms_per_slot_token']} | {fmt(r['rows_per_call'])} |")
    print()
    for label, arm in data.items():
        m = arm.get("markers") or {}
        raw = {name: {k: v for k, v in counts.items() if v} for name, counts in (m.get("raw_occurrences") or {}).items()}
        d = arm.get("decisions") or {}
        print(f"- {label}: CANDIDATE_REQUALIFIED decisions {m.get('candidate_requalified_decisions_by_model')}; "
              f"window ineligible reasons {(arm.get('window_detail') or {}).get('measurement_ineligible_reasons')}; "
              f"raw {raw}; lockout {arm.get('lockout')}; coherence host decisions "
              f"{d.get('coherence_decisions_fraction_zero')} {d.get('coherence_host_decisions_by_server_reason')}, "
              f"phone {d.get('coherence_decisions_phone')}")
    print()
    for label, arm in data.items():
        d = arm.get("decisions") or {}
        print(f"- {label} decisions: {d.get('by_model_reason_selected')}")
    print()
    print("| arm | request | model | out tokens | arrival s | acquired s | end s | phone calls |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for label, arm in data.items():
        for r in arm.get("request_intervals") or []:
            print(f"| {label} | {r['request']} | {r['model']} | {r['output_tokens']} | {r['arrival_s']} | "
                  f"{r['acquired_s']} | {r['end_s']} | {r['phone_calls']} |")
    print()
    print("| arm | request | model | windows | eligible (phone/host) | batches | host J/t | phone J/t | final ppm | "
          "host despite better phone | top decisions |")
    print("| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- |")
    for label, arm in data.items():
        for r in arm.get("requests") or []:
            print(f"| {label} | {r['request']} | {r['model']} | {r['windows']} | {r['eligible_windows']} "
                  f"({r['eligible_phone_windows']}/{r['eligible_host_windows']}) | {r['active_batches']} | "
                  f"{r['host_j_per_token']} | {r['phone_j_per_token']} | {r['final_fraction_ppm']} | "
                  f"{'YES' if r['host_despite_better_phone'] else ''} | {'; '.join(r['top_decisions'])} |")
    for pair in pairs:
        label, _, p = pair.partition("=")
        r = json.load(open(p))
        print(f"- exactness {label}: {r['status']} identical {r['identical_outputs']}; differences "
              f"{r['output_differences']}; host saving {r['host_saving_percent']:.2f} %, duration "
              f"{r['duration_change_percent']:+.2f} %")


if __name__ == "__main__":
    main(*sys.argv[1:])
