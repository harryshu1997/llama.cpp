#!/usr/bin/env python3
"""S10-V0 shared model: instance loading, deterministic non-preemptive schedule
simulation, and the total-wall energy model. Used by the oracle (oracle.py) and by
the policies (policies/policies.py). It is NOT imported by the certificate checker
(checker/checker.py), which re-derives feasibility and energy independently.

Units are exact integers: time in microseconds (us), power in milliwatts (mW),
energy in nanojoules (nJ = mW*us). No RNG and no wall clock here (determinism).

Model
-----
Each request r has a linear DAG: PRE_r (server) -> MID_r (FFN island) -> SUF_r
(server). PRE/SUF are individual server ops of latency pre_us/suf_us. MID_r is the
measured gemma4 dense-FFN island with mid_tokens tokens (M). A MID may run on the
SERVER (batchable with same-(wave,weight_set) server MIDs) or, if its weight_set is
resident and M is a phone-certified value, on a phone (one island per phone lane,
serial). Server is one non-preemptive machine; each phone is one non-preemptive lane.

Energy (per named power model, all values are DATA read from the instance):
  server_energy = P_gpu_active * server_busy_us   [+ P_gpu_idle*(H-busy) if count_idle]
  phone_energy  = sum_islands (P_phone + P_usb) * e2e_us[phone]   [+ idle if count_idle]
A schedule's assignment+batching fully determine energy (order-independent).
"""
import json


def load_instance(path):
    with open(path) as f:
        inst = json.load(f)
    # expand the request list into node structures
    return inst


def lffn_us(inst, m):
    """Look up server FFN batch latency for m tokens from the DATA table."""
    if m <= 0:
        return 0
    t = inst["lffn_us_table"]
    key = str(m)
    if key in t:
        return t[key]
    # clamp to table max (instances are bounded so this is only a safety net)
    return t[str(inst["lffn_us_table_max_m"])]


def phone_can_run(inst, phone, weight_set, mid_tokens):
    p = inst["phones"][phone]
    return (weight_set in p["resident_weight_sets"]) and (mid_tokens in p["certified_tokens"])


class Nodes:
    """Materialize PRE/MID/SUF nodes from the frozen requests."""
    def __init__(self, inst):
        self.inst = inst
        self.pre = {}   # r -> dict
        self.mid = {}
        self.suf = {}
        for r in inst["requests"]:
            rid = r["id"]
            self.pre[rid] = {"id": f"PRE_{rid}", "req": rid, "release": r["release_us"],
                             "lat": r.get("pre_us", inst["pre_us"]), "wave": r["wave"]}
            self.mid[rid] = {"id": f"MID_{rid}", "req": rid, "weight_set": r["weight_set"],
                             "tokens": r["mid_tokens"], "wave": r["wave"], "model_id": r["model_id"]}
            self.suf[rid] = {"id": f"SUF_{rid}", "req": rid, "lat": r.get("suf_us", inst["suf_us"]),
                             "deadline": r["deadline_us"]}


def simulate(inst, placement, batch_split):
    """Deterministic list-schedule simulator.

    placement: dict rid -> "SERVER" | "OP15" | "OP12"
    batch_split: dict (wave, weight_set) -> bool  (True = run each server MID of that
                 group individually; False = batch them into one GEMM)

    Returns a dict with per-node start/finish, batches, energy, and SLO stats, or
    raises ValueError if the placement is structurally illegal (phone route not
    certified/resident).
    Determinism: server picks the ready op with the earliest deadline, ties by id.
    """
    N = Nodes(inst)
    H = inst["horizon_us"]
    pm = inst["power_model"]

    # validate placement legality
    for rid, dev in placement.items():
        if dev != "SERVER":
            r = next(x for x in inst["requests"] if x["id"] == rid)
            if not phone_can_run(inst, dev, r["weight_set"], r["mid_tokens"]):
                raise ValueError(f"illegal phone route {dev} for {rid}")

    # ---- Build server MID batches ----
    server_mids = [rid for rid in N.mid if placement[rid] == "SERVER"]
    groups = {}
    for rid in server_mids:
        m = N.mid[rid]
        groups.setdefault((m["wave"], m["weight_set"]), []).append(rid)
    batches = []  # each: {"members":[rid...], "tokens":m, "lat":us}
    for key, members in groups.items():
        split = batch_split.get(f"{key[0]}|{key[1]}", False)
        if split:
            for rid in members:
                tk = N.mid[rid]["tokens"]
                batches.append({"members": [rid], "tokens": tk, "lat": lffn_us(inst, tk),
                                "wave": key[0], "weight_set": key[1]})
        else:
            tk = sum(N.mid[rid]["tokens"] for rid in members)
            batches.append({"members": members, "tokens": tk, "lat": lffn_us(inst, tk),
                            "wave": key[0], "weight_set": key[1]})
    mid_to_batch = {}
    for bi, b in enumerate(batches):
        for rid in b["members"]:
            mid_to_batch[rid] = bi

    # ---- Phone islands ----
    phone_islands = {"OP15": [], "OP12": []}
    for rid in N.mid:
        if placement[rid] != "SERVER":
            phone_islands[placement[rid]].append(rid)
    # phone runs its islands in FIFO by (release, id)
    for p in phone_islands:
        phone_islands[p].sort(key=lambda rid: (N.pre[rid]["release"], rid))

    # ---- Non-preemptive event simulation ----
    # server ops: PRE nodes, MID batches, SUF nodes. Single machine.
    finish = {}         # node id -> finish us
    start = {}
    deadline_of = {rid: N.suf[rid]["deadline"] for rid in N.suf}

    # ready predicates depend on finishes; do a simple time-stepping list scheduler.
    server_time = 0
    phone_time = {"OP15": 0, "OP12": 0}

    pending_pre = set(N.pre.keys())
    pending_batch = set(range(len(batches)))
    pending_suf = set(N.suf.keys())
    pending_phone = {p: list(phone_islands[p]) for p in phone_islands}

    # schedule phones greedily (independent lanes)
    def try_schedule_phones():
        for p in ("OP15", "OP12"):
            queue = pending_phone[p]
            i = 0
            while i < len(queue):
                rid = queue[i]
                pre_id = N.pre[rid]["id"]
                if pre_id in finish:
                    s = max(phone_time[p], finish[pre_id])
                    e2e = inst["phones"][p]["e2e_us"]
                    start[N.mid[rid]["id"]] = s
                    finish[N.mid[rid]["id"]] = s + e2e
                    phone_time[p] = s + e2e
                    queue.pop(i)
                else:
                    i += 1

    guard = 0
    maxsteps = 10 * (len(N.pre) + len(batches) + len(N.suf)) + 10
    while (pending_pre or pending_batch or pending_suf) and guard < maxsteps:
        guard += 1
        try_schedule_phones()
        # gather ready server ops
        ready = []  # (deadline, tie, kind, key)
        for rid in pending_pre:
            rel = N.pre[rid]["release"]
            if rel <= max(server_time, rel):
                ready.append((deadline_of[rid], f"PRE_{rid}", "pre", rid, max(server_time, rel)))
        for bi in pending_batch:
            b = batches[bi]
            if all(N.pre[m]["id"] in finish for m in b["members"]):
                est = max(server_time, max(finish[N.pre[m]["id"]] for m in b["members"]))
                dl = min(deadline_of[m] for m in b["members"])
                ready.append((dl, f"BATCH_{bi}", "batch", bi, est))
        for rid in pending_suf:
            mid_id = N.mid[rid]["id"]
            if mid_id in finish:
                est = max(server_time, finish[mid_id])
                ready.append((deadline_of[rid], f"SUF_{rid}", "suf", rid, est))
        if not ready:
            # advance server_time to the next phone/mid finish to unblock SUFs
            future = [finish[N.mid[rid]["id"]] for rid in N.mid
                      if N.mid[rid]["id"] in finish and any(rid in pending_suf for _ in [0])]
            future = [f for rid in pending_suf if (N.mid[rid]["id"] in finish)
                      for f in [finish[N.mid[rid]["id"]]]]
            if future:
                server_time = max(server_time, min(future))
                continue
            # nothing schedulable and nothing pending to unblock: stuck (shouldn't happen)
            break
        # earliest deadline first; tie by earliest start then id
        ready.sort(key=lambda x: (x[0], x[4], x[1]))
        dl, tie, kind, key, est = ready[0]
        if kind == "pre":
            rid = key
            s = est
            start[N.pre[rid]["id"]] = s
            finish[N.pre[rid]["id"]] = s + N.pre[rid]["lat"]
            server_time = s + N.pre[rid]["lat"]
            pending_pre.discard(rid)
        elif kind == "batch":
            bi = key
            b = batches[bi]
            s = est
            start[f"BATCH_{bi}"] = s
            fin = s + b["lat"]
            server_time = fin
            for m in b["members"]:
                finish[N.mid[m]["id"]] = fin
                start[N.mid[m]["id"]] = s
            pending_batch.discard(bi)
        else:  # suf
            rid = key
            s = est
            start[N.suf[rid]["id"]] = s
            finish[N.suf[rid]["id"]] = s + N.suf[rid]["lat"]
            server_time = s + N.suf[rid]["lat"]
            pending_suf.discard(rid)

    # ---- terminal outcomes ----
    completions = {}
    tardy = 0
    timeout = 0
    for rid in N.suf:
        sid = N.suf[rid]["id"]
        if sid not in finish or finish[sid] > H:
            completions[rid] = {"outcome": "TIMEOUT", "finish": finish.get(sid)}
            timeout += 1
        else:
            met = finish[sid] <= deadline_of[rid]
            completions[rid] = {"outcome": "MET" if met else "TARDY", "finish": finish[sid]}
            if not met:
                tardy += 1

    # ---- energy ----
    server_busy = 0
    for rid in N.pre:
        server_busy += N.pre[rid]["lat"]
    for rid in N.suf:
        server_busy += N.suf[rid]["lat"]
    for b in batches:
        server_busy += b["lat"]
    phone_busy = {"OP15": 0, "OP12": 0}
    for p in phone_islands:
        phone_busy[p] = len(phone_islands[p]) * inst["phones"][p]["e2e_us"]

    e_server = pm["gpu_active_mw"] * server_busy
    if pm.get("count_server_idle", False):
        e_server += pm["gpu_idle_mw"] * max(0, H - server_busy)
    e_phone = 0
    for p in phone_islands:
        e_phone += (pm["phone_mw"][p] + pm["usb_host_mw"]) * phone_busy[p]
        if pm.get("count_phone_idle", False):
            e_phone += pm.get("phone_idle_mw", 0) * max(0, H - phone_busy[p])
    total_e = e_server + e_phone

    return {
        "placement": placement,
        "batches": batches,
        "phone_islands": phone_islands,
        "start": start,
        "finish": finish,
        "completions": completions,
        "tardy": tardy,
        "timeout": timeout,
        "met": sum(1 for c in completions.values() if c["outcome"] == "MET"),
        "server_busy_us": server_busy,
        "phone_busy_us": phone_busy,
        "energy_nj": total_e,
        "energy_server_nj": e_server,
        "energy_phone_nj": e_phone,
    }


def objective_key(sim):
    """Lexicographic objective (lower is better):
       L1 (tardy+timeout), L2 total lateness proxy (timeout heavier), L3 -met,
       L4 energy. Returns a tuple for comparison."""
    return (sim["tardy"] + sim["timeout"], sim["timeout"], -sim["met"], sim["energy_nj"])
