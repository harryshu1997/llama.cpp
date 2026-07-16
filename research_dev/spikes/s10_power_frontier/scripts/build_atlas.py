#!/usr/bin/env python3
# S10-V0 CP2: assemble the minimum measured route/power atlas from raw artifacts.
# Every value is tagged evidence = measured | inferred | blocked with a provenance
# pointer. Inferred power values are chosen MECHANISM-FAVORABLE (they maximize the
# GPU energy a phone offload can save and minimize the phone/USB/host cost) so that
# a FAIL verdict is conservative: if the mechanism cannot win even when the unknown
# quantities are set in its favor, it cannot win. Deterministic (no RNG, no clock).
import json, os, sys, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")

def load_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def load_json(p):
    with open(p) as f:
        return json.load(f)

# --- MEASURED: A6000 FFN island latency vs batch M (this session) ---
lat = load_jsonl(os.path.join(ART, "cp2_a6000_ffn_latency.jsonl"))
a6000_lat = {r["M"]: {"compute_us_p50": r["compute_us_p50"],
                       "compute_us_p05": r["compute_us_p05"],
                       "compute_us_p95": r["compute_us_p95"],
                       "e2e_us_p50": r["e2e_us_p50"]} for r in lat}
n_embd = lat[0]["n_embd"]; n_ff = lat[0]["n_ff"]; weight_bytes = lat[0]["weight_bytes"]

# --- MEASURED: A6000 board power (this session) ---
pw_rows = []
with open(os.path.join(ART, "cp2_a6000_power_trace.csv")) as f:
    for line in f:
        p = line.split(",")
        if len(p) < 5:
            continue
        try:
            pw_rows.append((float(p[1]), float(p[4])))  # (power_W, util%)
        except ValueError:
            pass
busy = [w for w, u in pw_rows if u > 50]
idle = [w for w, u in pw_rows if u < 5]
a6000_power = {
    "idle_pstate": "P8",
    "idle_power_w": {"value": round(min(idle), 1), "evidence": "measured",
                     "prov": "cp2_a6000_power_trace.csv util<5 min; cp0_gpu_power_probe.txt ~25W steady"},
    "active_power_w_sustained_mean": {"value": round(st.mean(busy), 1), "evidence": "measured",
                     "prov": "cp2_a6000_power_trace.csv util>50 mean (back-to-back FFN)"},
    "active_power_w_ceiling": {"value": round(max(busy), 1), "evidence": "measured",
                     "prov": "cp2_a6000_power_trace.csv util>50 max (near 300W cap)"},
    "power_cap_min_w": 100, "power_cap_max_w": 300, "power_cap_default_w": 300,
    "power_cap_settable": {"value": False, "evidence": "measured",
                     "prov": "cp0: nvidia-smi -pl requires root; no passwordless sudo"},
    "lower_state_below_idle": {"value": None, "evidence": "measured",
                     "prov": "P8 (~25W) auto-entered is the floor; no scheduler-triggerable deeper sleep"},
    "sample_rate_hz": {"value": 1.5, "evidence": "measured",
                     "prov": "nvidia-smi internal 119 samples / 80.87 s; too coarse for few-ms islands"},
}

# --- MEASURED: phone FFN island (blk.2, M=16) from CP0 ---
op15 = load_json(os.path.join(ART, "cp0_ffn_op15.json"))
op12 = load_json(os.path.join(ART, "cp0_ffn_op12.json"))
def phone_entry(d, htp):
    return {
        "htp": htp,
        "compute_ms_p50": {"value": d["phone_compute_p50_ms"], "evidence": "measured", "prov": "cp0 prestaged FFN"},
        "transport_ms_p50": {"value": d["transport_and_protocol_p50_ms"], "evidence": "measured", "prov": "cp0"},
        "e2e_ms_p50": {"value": d["warm_e2e_p50_ms"], "evidence": "measured", "prov": "cp0 warm e2e"},
        "e2e_ms_p95": {"value": d["warm_e2e_p95_ms"], "evidence": "measured", "prov": "cp0 warm e2e"},
        "rel_l2_max": {"value": d["rel_l2_max"], "evidence": "measured", "prov": "cp0 vs production_gemma4_cb_eval"},
        "M": d["M"],
    }
phones = {"OP15": phone_entry(op15, "v81"), "OP12": phone_entry(op12, "v75")}

# --- MEASURED: boundary bytes per island (activation f32) ---
def boundary_bytes(M):
    return M * n_embd * 4
boundary = {"n_embd": n_embd, "n_ff": n_ff, "weight_bytes_f16": weight_bytes,
            "activation_in_bytes_per_M": n_embd * 4, "activation_out_bytes_per_M": n_embd * 4,
            "example_M16_in_bytes": boundary_bytes(16), "evidence": "measured",
            "prov": "cp2 harness input/output_bytes; cp0 n_embd/n_ff"}

# --- INFERRED (mechanism-favorable), labeled ---
inferred = {
    "phone_soc_power_w": {"value": 2.0, "evidence": "inferred_favorable",
        "prov": "prior spikes: USB rail ~1.7-1.8W near-const; SoC HTP compute plausibly 3-6W. "
                "We use the LOW end (2W) to MINIMIZE modeled phone cost and favor the mechanism.",
        "conservative_high_w": 6.0},
    "usb_host_relay_power_w": {"value": 0.0, "evidence": "inferred_favorable",
        "prov": "Set to 0 to give the mechanism the benefit of the doubt (ignore the cost of driving "
                "the phone over USB and the host relay CPU). Real value is > 0.",
        "conservative_high_w": 8.0},
    "a6000_transition_energy_j": {"value": 0.0, "evidence": "inferred_favorable",
        "prov": "P8<->P0 is a clock ramp; assume free instant transition to favor SLEEP/POWER bundles."},
    "a6000_wake_latency_ms": {"value": 0.0, "evidence": "inferred_favorable",
        "prov": "assume instant wake to favor the mechanism"},
    "a6000_gpu_active_power_w_for_energy": {"value": a6000_power["active_power_w_ceiling"]["value"],
        "evidence": "inferred_favorable",
        "prov": "use the MEASURED ceiling (~300W) as active power so a phone offload saves the MOST "
                "possible GPU board energy. Real bandwidth-bound FFN averages ~281W."},
}

# --- BLOCKED ---
blocked = {
    "complete_wall_energy": {"evidence": "blocked",
        "prov": "no synchronized external meter for host CPU/DRAM/PSU + A6000 + USB + phone/charger; "
                "only GPU board power available, at ~1.5 Hz (too coarse for few-ms islands). CP0 ENERGY_BLOCKED."},
    "phone_charger_vbus_power": {"evidence": "blocked",
        "prov": "USB rail pinned; battery Full->coulomb 0; no root OP12; no WiFi-adb"},
    "a6000_power_at_alternate_caps": {"evidence": "blocked",
        "prov": "cannot set -pl/-ac without root; power-vs-cap curve not measurable"},
}

# --- break-even gap (PLAN CP2 formula) ---
# t_break_even = wake_latency + transition_energy / (idle_power - lower_state_power)
# lower_state_power is undefined (no state below auto-P8) => term undefined/infinite.
break_even = {"defined": False, "evidence": "measured",
    "prov": "idle_power - lower_state_power undefined: no measured lower state below auto-P8 (25W). "
            "PLAN CP2: do not proceed to a physical energy gate when no measured lower state exists."}

# --- MEASURED->DERIVED: integer-us L_ffn(M) lookup table (piecewise-linear over
#     the measured points; extrapolated past 1024 with the compute-bound slope).
#     Stored as DATA so the oracle and the independent checker both look it up
#     rather than re-implementing interpolation logic. ---
mpts = sorted(a6000_lat.keys())
def lffn_us(M):
    if M <= 0:
        return 0
    if M in a6000_lat:
        return a6000_lat[M]["compute_us_p50"]
    if M < mpts[0]:
        return a6000_lat[mpts[0]]["compute_us_p50"]
    if M > mpts[-1]:
        a, b = mpts[-2], mpts[-1]
        slope = (a6000_lat[b]["compute_us_p50"] - a6000_lat[a]["compute_us_p50"]) / (b - a)
        return int(round(a6000_lat[b]["compute_us_p50"] + slope * (M - b)))
    for i in range(len(mpts) - 1):
        a, b = mpts[i], mpts[i + 1]
        if a <= M <= b:
            fa, fb = a6000_lat[a]["compute_us_p50"], a6000_lat[b]["compute_us_p50"]
            return int(round(fa + (fb - fa) * (M - a) / (b - a)))
    return a6000_lat[mpts[-1]]["compute_us_p50"]
MAX_M = 4096
lffn_table = {str(M): lffn_us(M) for M in range(0, MAX_M + 1)}

atlas = {
    "atlas_version": 1,
    "island": {"id": "gemma4_dense_ffn_blk2", "graph_hash_source": "phone_pim_ffn.cpp exact island",
               "n_embd": n_embd, "n_ff": n_ff, "weight_bytes_f16": weight_bytes},
    "a6000_ffn_latency_by_M_us": a6000_lat,
    "lffn_us_table": lffn_table,
    "lffn_us_table_max_m": MAX_M,
    "a6000_power": a6000_power,
    "phones": phones,
    "boundary_bytes": boundary,
    "inferred_favorable": inferred,
    "blocked": blocked,
    "break_even_gap": break_even,
    "energy_model_note": (
        "GPU energy over a schedule = idle_power*wall + (active_power-idle_power)*sum(busy island latency). "
        "Island latency uses the MEASURED a6000_ffn_latency_by_M_us curve, so batching (amortized 354MB "
        "weight read) is modeled exactly. The optimized server-only baseline C1 gets the SAME lazy batching "
        "and the 25W idle floor. Phone island energy = phone_soc_power*e2e + usb_host_relay_power*e2e."),
}

out = os.path.join(ART, "cp2_atlas.json")
with open(out, "w") as f:
    json.dump(atlas, f, indent=2, sort_keys=True)
print("wrote", out)
# quick echo of the decisive per-token numbers
def per_tok(M):
    us = a6000_lat[M]["compute_us_p50"]
    return us / M
print("A6000 FFN compute us/token: M16=%.2f M64=%.2f M256=%.2f M1024=%.2f"
      % (per_tok(16), per_tok(64), per_tok(256), per_tok(1024)))
print("A6000 marginal us/token M16->M256 (compute-bound slope): %.3f"
      % ((a6000_lat[256]["compute_us_p50"] - a6000_lat[16]["compute_us_p50"]) / (256 - 16)))
gpu_active = inferred["a6000_gpu_active_power_w_for_energy"]["value"]
print("A6000 batched marginal energy mJ/token (@%.0fW, M256 slope): %.4f"
      % (gpu_active, gpu_active * ((a6000_lat[256]["compute_us_p50"] - a6000_lat[16]["compute_us_p50"]) / (256-16)) / 1000.0))
print("Phone OP15 energy mJ/token (@2W, e2e 27.1ms, M16): %.4f" % (2.0 * op15["warm_e2e_p50_ms"] / op15["M"]))
print("Phone OP12 energy mJ/token (@2W, e2e 45.0ms, M16): %.4f" % (2.0 * op12["warm_e2e_p50_ms"] / op12["M"]))
