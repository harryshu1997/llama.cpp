#!/usr/bin/env python3
"""S10-V0 policies C0..C5. Each maps an instance to a (placement, batch_split) and
emits a certificate (built with oracle.build_certificate) that the INDEPENDENT
checker validates. Uses model_data (allowed); the checker shares no code.

  C0 eager server-only FIFO         : all SERVER, no batching (split-all)
  C1 optimized server-only          : all SERVER, energy-optimal batch/split (SLO-safe)   <-- BASELINE
  C2 fixed phone placement          : offload every phone-resident MID round-robin, no shaping
  C3 frontier shaping, no power credit: offload greedily whenever phone is SLO-safe (over-offload)
  C4 perfect-future Q-PIM oracle     : oracle.solve (min energy, SLO-feasible, sees all)
  C5 bounded causal Q-PIM            : per-wave beam over ARRIVED nodes only, commit, replan
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "oracle"))
import model_data as M          # noqa: E402
import oracle as ORACLE         # noqa: E402


def _server_groups(inst, placement):
    ks = set()
    for r in inst["requests"]:
        if placement[r["id"]] == "SERVER":
            ks.add((r["wave"], r["weight_set"]))
    return sorted(ks)


def _best_batch_split(inst, placement):
    """Given a placement, choose the batch/split per server group that minimizes the
    lexicographic objective (energy-optimal, SLO-safe). Groups are few -> enumerate."""
    gks = _server_groups(inst, placement)
    best = None
    best_key = None
    best_sim = None
    for bits in itertools.product([False, True], repeat=len(gks)):
        bs = {f"{gk[0]}|{gk[1]}": b for gk, b in zip(gks, bits)}
        try:
            sim = M.simulate(inst, placement, bs)
        except ValueError:
            continue
        key = M.objective_key(sim)
        if best_key is None or key < best_key:
            best_key, best, best_sim = key, bs, sim
    return best, best_sim


def policy_C0(inst):
    placement = {r["id"]: "SERVER" for r in inst["requests"]}
    gks = _server_groups(inst, placement)
    bs = {f"{gk[0]}|{gk[1]}": True for gk in gks}   # split-all = eager, no batching
    return placement, bs


def policy_C1(inst):
    placement = {r["id"]: "SERVER" for r in inst["requests"]}
    bs, _ = _best_batch_split(inst, placement)
    return placement, bs


def _phone_for(inst, r, load):
    """pick a resident phone (round-robin by a simple deterministic key)."""
    cands = [p for p in ("OP15", "OP12") if M.phone_can_run(inst, p, r["weight_set"], r["mid_tokens"])]
    if not cands:
        return None
    return cands[load % len(cands)]


def policy_C2(inst):
    """Fixed placement: offload EVERY phone-resident MID, round-robin, no shaping."""
    placement = {}
    k = 0
    for r in inst["requests"]:
        p = _phone_for(inst, r, k)
        if p is not None:
            placement[r["id"]] = p
            k += 1
        else:
            placement[r["id"]] = "SERVER"
    bs, _ = _best_batch_split(inst, placement)
    return placement, bs


def policy_C3(inst):
    """Frontier shaping without power-trigger credit: offload a MID to a phone
    whenever a phone lane is SLO-safe for it, regardless of whether it lowers total
    energy (i.e. shape the frontier greedily). Deterministic phone lane packing."""
    placement = {r["id"]: "SERVER" for r in inst["requests"]}
    lane_free = {"OP15": {}, "OP12": {}}   # phone -> wave -> next free time (approx)
    for r in sorted(inst["requests"], key=lambda x: (x["wave"], x["id"])):
        best_p = None
        for p in ("OP15", "OP12"):
            if not M.phone_can_run(inst, p, r["weight_set"], r["mid_tokens"]):
                continue
            # SLO-safe if pre + e2e + suf fits before deadline (optimistic frontier check)
            finish = r["release_us"] + inst["pre_us"] + inst["phones"][p]["e2e_us"] + r.get("suf_us", inst["suf_us"])
            if finish <= r["deadline_us"]:
                best_p = p
                break
        if best_p is not None:
            placement[r["id"]] = best_p
    bs, _ = _best_batch_split(inst, placement)
    return placement, bs


def policy_C4(inst, max_enum=2_000_000):
    placement, batch_split, sim, _, _ = ORACLE.solve(inst, max_enum)
    return placement, batch_split


def policy_C5(inst):
    """Bounded causal policy: process waves in arrival order. At each wave, with a
    bounded beam over ARRIVED phone-eligible MIDs only (no future arrivals), choose
    the per-wave placement minimizing the CURRENT-model causal objective subject to
    SLO and phone-lane capacity, commit, and move on. H-hop lookahead = 1 wave."""
    placement = {r["id"]: "SERVER" for r in inst["requests"]}
    waves = sorted({r["wave"] for r in inst["requests"]})
    for w in waves:
        wave_reqs = [r for r in inst["requests"] if r["wave"] == w]
        eligible = [r for r in wave_reqs
                    if any(M.phone_can_run(inst, p, r["weight_set"], r["mid_tokens"]) for p in ("OP15", "OP12"))]
        # beam over subsets is exponential; bound by per-class counts within the wave
        # (same symmetry the oracle uses), but restricted to arrived (this wave) work.
        classes = {}
        for r in eligible:
            classes.setdefault((r["weight_set"], r["deadline_us"], r["model_id"]), []).append(r["id"])
        # enumerate device-count compositions per class, pick best CURRENT-objective incumbent
        class_items = sorted(classes.items())
        best_key = None
        best_assign = None
        import itertools as _it
        comp_lists = []
        for _, members in class_items:
            g = len(members)
            devs = ["SERVER", "OP15", "OP12"]
            # restrict to resident phones for this class
            rep = next(r for r in eligible if r["id"] == members[0])
            devs = ["SERVER"] + [p for p in ("OP15", "OP12")
                                 if M.phone_can_run(inst, p, rep["weight_set"], rep["mid_tokens"])]
            comps = list(ORACLE.compositions(g, len(devs)))
            comp_lists.append((members, devs, comps))
        for combo in _it.product(*[c[2] for c in comp_lists]):
            trial = dict(placement)
            for (members, devs, _), counts in zip(comp_lists, combo):
                i = 0
                for dev, c in zip(devs, counts):
                    for _ in range(c):
                        trial[members[i]] = dev
                        i += 1
            bs, sim = _best_batch_split(inst, trial)
            if sim is None:
                continue
            key = M.objective_key(sim)
            if best_key is None or key < best_key:
                best_key, best_assign = key, dict(trial)
        if best_assign is not None:
            placement = best_assign
    bs, _ = _best_batch_split(inst, placement)
    return placement, bs


POLICIES = {"C0": policy_C0, "C1": policy_C1, "C2": policy_C2,
            "C3": policy_C3, "C4": policy_C4, "C5": policy_C5}


def run_policy(inst, name):
    placement, batch_split = POLICIES[name](inst)
    sim = M.simulate(inst, placement, batch_split)
    cert = ORACLE.build_certificate(inst, placement, batch_split, sim, label=name)
    return cert, sim


def timely_phone_islands(cert):
    n = 0
    for s in cert["schedule"]:
        if s["mid_device"] != "SERVER" and s["outcome"] == "MET":
            n += 1
    return n


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--policy", default="all")
    args = ap.parse_args()
    inst = M.load_instance(args.instance)
    names = list(POLICIES) if args.policy == "all" else [args.policy]
    out = {}
    for nm in names:
        cert, sim = run_policy(inst, nm)
        out[nm] = {"energy_nj": cert["energy_nj"], "met": cert["met"], "tardy": cert["tardy"],
                   "timeout": cert["timeout"], "timely_phone_islands": timely_phone_islands(cert),
                   "phone_offload": {p: v for p, v in cert["phone_islands"].items() if v}}
    print(json.dumps({"instance": inst["instance_id"], "policies": out}, indent=2, sort_keys=True))
