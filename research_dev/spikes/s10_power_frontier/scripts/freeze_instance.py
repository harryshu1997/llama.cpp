#!/usr/bin/env python3
"""S10-V0 CP1: freeze the controlled tiny multi-DAG instances.

Builds canonical instance JSON records from the CP2 measured atlas. Three DAG
templates (T_gen server-pre -> phone-FFN -> server-suffix; T_service stateless
server-pre -> phone-FFN; T_rag server-pre -> phone-FFN -> server-suffix on a second
weight set). At least two model_id and two weight_set values. mid_tokens fixed at 16
(the only phone-certified point). Load/burstiness/slack/weight-mix sweeps and the
phone thermal profile are explicit parameters. Deterministic (no RNG).

Outputs:
  fixtures/frozen/primary_favorable.json, primary_conservative.json  (representative)
  fixtures/frozen/sweep/*.json                                        (CP4 sweep grid)
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")
FROZEN = os.path.join(ROOT, "fixtures", "frozen")
SWEEP = os.path.join(FROZEN, "sweep")

ATLAS = json.load(open(os.path.join(ART, "cp2_atlas.json")))
LFFN = ATLAS["lffn_us_table"]
LFFN_MAX = ATLAS["lffn_us_table_max_m"]
E2E_OP15 = int(round(ATLAS["phones"]["OP15"]["e2e_ms_p50"]["value"] * 1000))   # us
E2E_OP12 = int(round(ATLAS["phones"]["OP12"]["e2e_ms_p50"]["value"] * 1000))   # us

POWER_FAVORABLE = {
    "name": "favorable",
    "gpu_active_mw": 300000, "gpu_idle_mw": 25000,
    "phone_mw": {"OP15": 2000, "OP12": 2000}, "usb_host_mw": 0,
    "count_server_idle": False, "count_phone_idle": False,
    "note": "mechanism-favorable: GPU 300W ceiling, phone 2W, USB/host 0W, idle uncounted",
}
POWER_CONSERVATIVE = {
    "name": "conservative",
    "gpu_active_mw": 281000, "gpu_idle_mw": 25000,
    "phone_mw": {"OP15": 6000, "OP12": 6000}, "usb_host_mw": 8000,
    "count_server_idle": True, "count_phone_idle": False,
    "note": "conservative: GPU 281W measured mean, phone 6W, USB/host 8W, server idle counted",
}

# thermal profile -> phone e2e multiplier (warm = measured 1.0; throttled from OP15 95C)
THERMAL = {"warm": 1.0, "throttled": 1.3, "cold": 1.1}


def make_instance(instance_id, *, waves, per_wave, weight_mix, slack_us, horizon_us,
                  power_model, thermal="warm", phone_resident=("w0", "w1"),
                  wave_spacing_us=8000, pre_us=50, suf_us=50):
    """weight_mix: list of (weight_set, model_id) assigned round-robin to requests;
    a mix with many distinct weight_sets => lone (unbatchable) islands, a mix with
    repeats => native server batches."""
    mult = THERMAL[thermal]
    phones = {
        "OP15": {"e2e_us": int(round(E2E_OP15 * mult)), "resident_weight_sets": list(phone_resident),
                 "certified_tokens": [16], "htp": "v81", "thermal": thermal},
        "OP12": {"e2e_us": int(round(E2E_OP12 * mult)), "resident_weight_sets": list(phone_resident),
                 "certified_tokens": [16], "htp": "v75", "thermal": thermal},
    }
    requests = []
    idx = 0
    weight_sets = sorted({w for w, _ in weight_mix})
    for w in range(waves):
        rel = w * wave_spacing_us
        for k in range(per_wave):
            ws, mid = weight_mix[idx % len(weight_mix)]
            has_suffix = (mid != "m_service")
            requests.append({
                "id": f"r{idx:02d}", "wave": w, "release_us": rel,
                "deadline_us": rel + slack_us, "weight_set": ws,
                "model_id": mid, "mid_tokens": 16,
                "pre_us": pre_us, "suf_us": (suf_us if has_suffix else 0),
                "template": "T_service" if mid == "m_service" else ("T_rag" if ws == "w1" else "T_gen"),
            })
            idx += 1
    inst = {
        "instance_version": 1,
        "instance_id": instance_id,
        "horizon_us": horizon_us,
        "pre_us": pre_us, "suf_us": suf_us,
        "activation_mem_bound_bytes": 64 * 1024 * 1024,
        "power_model": {k: v for k, v in power_model.items()},
        "phones": phones,
        "weight_sets": weight_sets,
        "model_ids": sorted({m for _, m in weight_mix}),
        "requests": requests,
        "lffn_us_table": LFFN,
        "lffn_us_table_max_m": LFFN_MAX,
        "atlas_provenance": {
            "island": ATLAS["island"]["id"], "n_embd": ATLAS["island"]["n_embd"],
            "n_ff": ATLAS["island"]["n_ff"], "weight_bytes_f16": ATLAS["island"]["weight_bytes_f16"],
            "op15_e2e_us_measured": E2E_OP15, "op12_e2e_us_measured": E2E_OP12,
            "thermal_multiplier": mult,
        },
        "cp1_params": {"waves": waves, "per_wave": per_wave, "slack_us": slack_us,
                       "thermal": thermal, "weight_mix": weight_mix,
                       "phone_resident": list(phone_resident), "wave_spacing_us": wave_spacing_us},
    }
    return inst


def dump(inst, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(inst, f, indent=2, sort_keys=True)


# ---- Representative primary instance (realistic mixed serving) ----
# 2 models m0(w0),m1(w1); m0-heavy with repeats (batchable), m1 present (2nd weight).
PRIMARY_MIX = [("w0", "m0"), ("w0", "m0"), ("w0", "m0"), ("w1", "m1"),
               ("w0", "m0"), ("w0", "m0")]  # 5x w0/m0, 1x w1/m1 per 6 -> repeats batch


def main():
    # primary representative: 2 waves x 6 = 12 requests, 60ms slack, warm phones
    for pm, tag in ((POWER_FAVORABLE, "favorable"), (POWER_CONSERVATIVE, "conservative")):
        inst = make_instance(f"primary_{tag}", waves=2, per_wave=6, weight_mix=PRIMARY_MIX,
                             slack_us=60000, horizon_us=80000, power_model=pm, thermal="warm")
        dump(inst, os.path.join(FROZEN, f"primary_{tag}.json"))

    # ---- CP4 sweep grid ----
    # load bins (per_wave) are per-mix so every C4 stays exactly enumerable. weight-mix
    # spans all-batchable -> realistic 2-model -> all-lone (unbatchable, favors offload).
    MIXES = {
        # (weight_mix, load bins) ; loads chosen so 3^(phone-eligible) is enumerable
        "batchable": ([("w0", "m0")], [2, 4, 6, 9]),                 # one weight -> big server batch
        "twomodel":  (PRIMARY_MIX, [2, 4, 6, 9]),                    # realistic 2-model mix
        "lone":      ([("w0", "m0"), ("w1", "m1"), ("w2", "m2"), ("w3", "m3")], [2, 3, 4]),  # distinct -> lone
    }
    SLACKS = {"tight": 30000, "slack": 90000}
    count = 0
    for pm, tag in ((POWER_FAVORABLE, "favorable"), (POWER_CONSERVATIVE, "conservative")):
        for mixname, (mix, loads) in MIXES.items():
            for load in loads:
                for sname, slack in SLACKS.items():
                    resident = tuple(sorted({w for w, _ in mix}))  # phones hold all weights (favorable)
                    inst = make_instance(
                        f"sweep_{tag}_{mixname}_load{load}_{sname}",
                        waves=2, per_wave=load, weight_mix=mix, slack_us=slack,
                        horizon_us=200000, power_model=pm, thermal="warm",
                        phone_resident=resident)
                    dump(inst, os.path.join(SWEEP, f"sweep_{tag}_{mixname}_load{load}_{sname}.json"))
                    count += 1
    print(f"froze primary (favorable, conservative) + {count} sweep instances")
    print(f"  phone e2e us: OP15={E2E_OP15} OP12={E2E_OP12}")


if __name__ == "__main__":
    main()
