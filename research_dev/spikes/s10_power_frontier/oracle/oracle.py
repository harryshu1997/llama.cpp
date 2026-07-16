#!/usr/bin/env python3
"""S10-V0 exact tiny oracle. Deterministic exhaustive enumeration over MID route
placements and per-group server batch/split choices, scored by the fixed
lexicographic objective (L1 SLO misses, L2 timeouts, L3 -met, L4 total wall energy).
Emits a canonical solution certificate (with a sha256) that the INDEPENDENT checker
(checker/checker.py) validates. No RNG, no wall clock.

Usage:
  oracle.py --instance INST.json [--out CERT.json] [--max-enum N] [--label C4_oracle]
Exit codes: 0 optimum found; 2 usage/enumeration-bound error (fail closed).
"""
import argparse
import hashlib
import itertools
import json
import sys

import model_data as M


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def sha256_hex(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def request_class(r):
    """Interchangeability key: requests with identical class are symmetric, so the
    oracle enumerates device COUNTS per class instead of per request. This is exact
    (energy is count-determined; phone lanes are FIFO over identical deadlines)."""
    return (r["wave"], r["weight_set"], r["deadline_us"], r.get("pre_us"), r.get("suf_us"),
            r["model_id"], r["mid_tokens"])


def legal_devices(inst, r):
    opts = ["SERVER"]
    for p in ("OP15", "OP12"):
        if M.phone_can_run(inst, p, r["weight_set"], r["mid_tokens"]):
            opts.append(p)
    return opts


def compositions(g, k):
    """All ways to write integer g as an ordered sum of k non-negative integers."""
    if k == 1:
        yield (g,)
        return
    for first in range(g + 1):
        for rest in compositions(g - first, k - 1):
            yield (first,) + rest


def group_keys(inst, placement):
    ks = set()
    for r in inst["requests"]:
        if placement[r["id"]] == "SERVER":
            ks.add((r["wave"], r["weight_set"]))
    return sorted(ks)


def solve(inst, max_enum=2_000_000):
    # symmetry-reduced enumeration over per-class device-count compositions
    classes = {}
    for r in inst["requests"]:
        classes.setdefault(request_class(r), []).append(r["id"])
    class_keys = sorted(classes.keys())
    class_devs = []
    class_comps = []
    total = 1
    for ck in class_keys:
        members = sorted(classes[ck])
        rep = next(r for r in inst["requests"] if r["id"] == members[0])
        devs = legal_devices(inst, rep)
        comps = list(compositions(len(members), len(devs)))
        class_devs.append((members, devs))
        class_comps.append(comps)
        total *= len(comps)
    if total > max_enum:
        raise RuntimeError(f"reduced placement space {total} exceeds max_enum {max_enum} (fail closed); "
                           f"shrink instance or raise --max-enum deliberately")

    best = None
    best_key = None
    best_sim = None
    n_eval = 0
    for combo in itertools.product(*class_comps):
        placement = {}
        for (members, devs), counts in zip(class_devs, combo):
            i = 0
            for dev, c in zip(devs, counts):
                for _ in range(c):
                    placement[members[i]] = dev
                    i += 1
        gks = group_keys(inst, placement)
        # enumerate batch/split per group (2^len). groups are few for tiny instances.
        if len(gks) > 20:
            raise RuntimeError("too many server batch groups to enumerate (fail closed)")
        for bits in itertools.product([False, True], repeat=len(gks)):
            batch_split = {f"{gk[0]}|{gk[1]}": b for gk, b in zip(gks, bits)}
            try:
                sim = M.simulate(inst, placement, batch_split)
            except ValueError:
                continue
            n_eval += 1
            key = M.objective_key(sim)
            if best_key is None or key < best_key:
                best_key = key
                best = (placement, batch_split)
                best_sim = sim
    if best is None:
        raise RuntimeError("no feasible schedule enumerated (fail closed)")
    return best[0], best[1], best_sim, n_eval, total


def build_certificate(inst, placement, batch_split, sim, label):
    # explicit per-node schedule so the checker can validate without re-solving
    schedule = []
    for r in inst["requests"]:
        rid = r["id"]
        schedule.append({
            "request": rid,
            "model_id": r["model_id"],
            "weight_set": r["weight_set"],
            "mid_tokens": r["mid_tokens"],
            "mid_device": placement[rid],
            "pre_start": sim["start"][f"PRE_{rid}"], "pre_finish": sim["finish"][f"PRE_{rid}"],
            "mid_start": sim["start"][f"MID_{rid}"], "mid_finish": sim["finish"][f"MID_{rid}"],
            "suf_start": sim["start"][f"SUF_{rid}"], "suf_finish": sim["finish"][f"SUF_{rid}"],
            "deadline": r["deadline_us"],
            "outcome": sim["completions"][rid]["outcome"],
        })
    batches = []
    for bi, b in enumerate(sim["batches"]):
        batches.append({"batch_id": bi, "wave": b["wave"], "weight_set": b["weight_set"],
                        "members": sorted(b["members"]), "tokens": b["tokens"],
                        "latency_us": b["lat"], "start": sim["start"][f"BATCH_{bi}"],
                        "finish": sim["start"][f"BATCH_{bi}"] + b["lat"]})
    cert = {
        "certificate_version": 1,
        "label": label,
        "instance_id": inst["instance_id"],
        "instance_sha256": sha256_hex(inst),
        "power_model": inst["power_model"],
        "schedule": schedule,
        "server_batches": batches,
        "phone_islands": {p: sorted(sim["phone_islands"][p]) for p in sim["phone_islands"]},
        "hbm": {"mode": {ws: "server" for ws in inst["weight_sets"]}, "relief_bytes": 0},
        "energy_nj": sim["energy_nj"],
        "energy_server_nj": sim["energy_server_nj"],
        "energy_phone_nj": sim["energy_phone_nj"],
        "server_busy_us": sim["server_busy_us"],
        "phone_busy_us": sim["phone_busy_us"],
        "tardy": sim["tardy"], "timeout": sim["timeout"], "met": sim["met"],
        "objective_key": list(M.objective_key(sim)),
    }
    cert["certificate_sha256"] = sha256_hex({k: cert[k] for k in cert if k != "certificate_sha256"})
    return cert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--out")
    ap.add_argument("--label", default="C4_oracle")
    ap.add_argument("--max-enum", type=int, default=2_000_000)
    args = ap.parse_args()
    inst = M.load_instance(args.instance)
    try:
        placement, batch_split, sim, n_eval, total = solve(inst, args.max_enum)
    except RuntimeError as e:
        print(f"ORACLE_FAIL: {e}", file=sys.stderr)
        return 2
    cert = build_certificate(inst, placement, batch_split, sim, args.label)
    js = json.dumps(cert, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w") as f:
            f.write(js + "\n")
    print(json.dumps({"instance": inst["instance_id"], "evaluated": n_eval, "space": total,
                      "energy_nj": cert["energy_nj"], "tardy": cert["tardy"],
                      "timeout": cert["timeout"], "met": cert["met"],
                      "cert_sha256": cert["certificate_sha256"],
                      "out": args.out}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
