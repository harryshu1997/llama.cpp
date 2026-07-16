#!/usr/bin/env python3
"""S10-V0 CP3: >=1000 deterministic generated tiny fixtures. For each seed we build
a small random instance (bounded so the oracle enumerates exactly), solve it with the
oracle, and require the INDEPENDENT checker to (a) ACCEPT the oracle certificate and
(b) REJECT a deterministically corrupted copy. Fully deterministic: seeds drive a
local random.Random; no wall clock. Exits nonzero if any oracle cert is rejected or
any corruption is accepted (fail-closed).

Usage: gen_fixtures.py [--n 1200] [--out fixtures_summary.jsonl]
"""
import argparse
import copy
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SP, "oracle"))
sys.path.insert(0, os.path.join(SP, "checker"))
import model_data as M      # noqa: E402
import oracle as ORACLE     # noqa: E402
import checker as CHK       # noqa: E402

ATLAS = json.load(open(os.path.join(SP, "artifacts", "cp2_atlas.json")))
LFFN = ATLAS["lffn_us_table"]
LFFN_MAX = ATLAS["lffn_us_table_max_m"]
E2E = {"OP15": int(round(ATLAS["phones"]["OP15"]["e2e_ms_p50"]["value"] * 1000)),
       "OP12": int(round(ATLAS["phones"]["OP12"]["e2e_ms_p50"]["value"] * 1000))}
WBYTES = ATLAS["island"]["weight_bytes_f16"]
N_EMBD = ATLAS["island"]["n_embd"]
N_FF = ATLAS["island"]["n_ff"]

POWER = {
    "favorable": {"name": "favorable", "gpu_active_mw": 300000, "gpu_idle_mw": 25000,
                  "phone_mw": {"OP15": 2000, "OP12": 2000}, "usb_host_mw": 0,
                  "count_server_idle": False, "count_phone_idle": False},
    "conservative": {"name": "conservative", "gpu_active_mw": 281000, "gpu_idle_mw": 25000,
                     "phone_mw": {"OP15": 6000, "OP12": 6000}, "usb_host_mw": 8000,
                     "count_server_idle": True, "count_phone_idle": False},
}


def gen_instance(seed):
    rng = random.Random(seed)
    n_weights = rng.randint(1, 3)
    weights = [f"w{i}" for i in range(n_weights)]
    resident = [w for w in weights if rng.random() < 0.7] or [weights[0]]
    waves = rng.randint(1, 2)
    per_wave = rng.randint(1, 4)
    slack = rng.choice([30000, 60000, 90000, 120000])
    pmname = rng.choice(["favorable", "conservative"])
    thermal_mult = rng.choice([1.0, 1.1, 1.3])
    e2e = {p: int(round(E2E[p] * thermal_mult)) for p in E2E}
    spacing = rng.choice([6000, 8000, 12000])
    pre_us = rng.choice([30, 50, 80])
    suf_us = rng.choice([0, 40, 60])
    reqs = []
    idx = 0
    for w in range(waves):
        rel = w * spacing
        for _ in range(per_wave):
            ws = rng.choice(weights)
            reqs.append({"id": f"r{idx:02d}", "wave": w, "release_us": rel,
                         "deadline_us": rel + slack, "weight_set": ws,
                         "model_id": f"m_{ws}", "mid_tokens": 16,
                         "pre_us": pre_us, "suf_us": suf_us, "template": "T_gen"})
            idx += 1
    inst = {
        "instance_version": 1, "instance_id": f"gen_{seed:05d}",
        "horizon_us": max(r["deadline_us"] for r in reqs) + 40000,
        "pre_us": pre_us, "suf_us": suf_us, "activation_mem_bound_bytes": 64 * 1024 * 1024,
        "power_model": copy.deepcopy(POWER[pmname]),
        "phones": {p: {"e2e_us": e2e[p], "resident_weight_sets": list(resident),
                       "certified_tokens": [16], "htp": "vX", "thermal": "gen"} for p in E2E},
        "weight_sets": sorted(set(weights)),
        "model_ids": sorted({f"m_{w}" for w in weights}),
        "requests": reqs,
        "lffn_us_table": LFFN, "lffn_us_table_max_m": LFFN_MAX,
        "atlas_provenance": {"island": "gemma4_dense_ffn_blk2", "n_embd": N_EMBD, "n_ff": N_FF,
                             "weight_bytes_f16": WBYTES, "op15_e2e_us_measured": E2E["OP15"],
                             "op12_e2e_us_measured": E2E["OP12"], "thermal_multiplier": thermal_mult},
        "cp1_params": {"waves": waves, "per_wave": per_wave, "slack_us": slack,
                       "thermal": "gen", "weight_mix": [[w, f"m_{w}"] for w in weights],
                       "phone_resident": resident, "wave_spacing_us": spacing},
    }
    return inst


def corrupt(cert, seed):
    """Deterministically corrupt a certificate in one of several ways; the checker
    must reject every corruption."""
    rng = random.Random(seed * 2654435761 & 0xffffffff)
    c = copy.deepcopy(cert)
    kind = rng.randint(0, 4)
    if kind == 0:
        c["energy_nj"] += rng.randint(1, 1000)
    elif kind == 1 and c["schedule"]:
        s = rng.choice(c["schedule"]); s["outcome"] = "MET" if s["outcome"] != "MET" else "TARDY"
    elif kind == 2 and c["schedule"]:
        s = rng.choice(c["schedule"]); s["mid_start"] = s["pre_finish"] - 1
    elif kind == 3:
        c["tardy"] = c["tardy"] + 1
    else:
        c["energy_phone_nj"] = c.get("energy_phone_nj", 0) + rng.randint(1, 1000)
    # reseal so the checker must catch the SUBSTANTIVE issue, not the hash
    body = {k: c[k] for k in c if k != "certificate_sha256"}
    c["certificate_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return c, kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--out", default=os.path.join(SP, "artifacts", "cp3_fixtures_summary.jsonl"))
    ap.add_argument("--examples", type=int, default=5)
    args = ap.parse_args()

    n_ok = n_reject_ok = 0
    accepted_corruption = 0
    rejected_valid = 0
    ex_dir = os.path.join(HERE, "generated_examples")
    os.makedirs(ex_dir, exist_ok=True)
    with open(args.out, "w") as fout:
        for seed in range(args.n):
            inst = gen_instance(seed)
            try:
                placement, batch_split, sim, _, _ = ORACLE.solve(inst, max_enum=2_000_000)
            except RuntimeError as e:
                # instance too large to enumerate exactly -> skip (fail closed: never
                # counts as a pass); should not happen for these bounds
                fout.write(json.dumps({"seed": seed, "skipped": str(e)}) + "\n")
                continue
            cert = ORACLE.build_certificate(inst, placement, batch_split, sim, "C4_oracle")
            fails = CHK.check(inst, cert)
            if fails:
                rejected_valid += 1
                fout.write(json.dumps({"seed": seed, "VALID_CERT_REJECTED": fails[:2]}) + "\n")
                continue
            n_ok += 1
            cc, kind = corrupt(cert, seed)
            cfails = CHK.check(inst, cc)
            if cfails:
                n_reject_ok += 1
            else:
                accepted_corruption += 1
            fout.write(json.dumps({"seed": seed, "cert_sha256": cert["certificate_sha256"],
                                   "energy_nj": cert["energy_nj"], "valid": True,
                                   "corruption_kind": kind, "corruption_rejected": bool(cfails)}) + "\n")
            if seed < args.examples:
                with open(os.path.join(ex_dir, f"gen_{seed:05d}_instance.json"), "w") as f:
                    json.dump(inst, f, indent=2, sort_keys=True)
                with open(os.path.join(ex_dir, f"gen_{seed:05d}_cert.json"), "w") as f:
                    json.dump(cert, f, indent=2, sort_keys=True)
    print(json.dumps({
        "generated": args.n, "oracle_certs_valid": n_ok,
        "valid_certs_wrongly_rejected": rejected_valid,
        "corruptions_correctly_rejected": n_reject_ok,
        "corruptions_wrongly_accepted": accepted_corruption,
        "summary": args.out,
    }, sort_keys=True))
    return 0 if (rejected_valid == 0 and accepted_corruption == 0 and n_ok >= 1000) else 2


if __name__ == "__main__":
    sys.exit(main())
