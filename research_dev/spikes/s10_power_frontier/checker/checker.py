#!/usr/bin/env python3
"""S10-V0 standalone solution-certificate checker.

Independent of the oracle, the policies, and the simulator (imports none of them).
It reads only the instance JSON and a certificate JSON and re-derives every
feasibility constraint and the total-wall energy from first principles, failing
closed (nonzero exit) on any violation. Shared inputs are DATA only (the L_ffn
table, phone e2e, power model, all inside the instance).

Validated:
  * instance_sha256 and certificate_sha256 integrity
  * exactly one terminal outcome per request; no duplicate/stale completion
  * precedence + READY-before-run: pre_start>=release; mid_start>=pre_finish;
    suf_start>=mid_finish (D2H before the server suffix)
  * fixed op latencies (pre_us, suf_us); server MID batch latency == L_ffn(sum tokens)
  * server single machine: no two server ops overlap
  * phone lane single: no two islands on a phone overlap; phone route resident+certified
  * batch legality: members share (wave, weight_set); tokens==sum; one batch per MID
  * energy recomputation (server active*busy [+idle]; phone (phone+usb)*e2e [+idle])
  * outcome correctness vs deadline and horizon; counts match
  * activation-memory peak <= bound
  * HBM relief credited only for EXCLUSIVE weights absent from the server (mirrored=0)

Usage: checker.py --instance INST.json --certificate CERT.json
Exit: 0 valid; 2 invalid (prints FAIL: reasons); 3 usage.
"""
import argparse
import hashlib
import json
import sys


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def sha256_hex(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def lffn(inst, m):
    if m <= 0:
        return 0
    t = inst["lffn_us_table"]
    return t.get(str(m), t[str(inst["lffn_us_table_max_m"])])


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def check(inst, cert):
    fails = []

    def req(cond, msg):
        if not cond:
            fails.append(msg)

    # ---- integrity ----
    req(cert.get("instance_sha256") == sha256_hex(inst),
        "instance_sha256 mismatch (certificate not bound to this instance)")
    body = {k: cert[k] for k in cert if k != "certificate_sha256"}
    req(cert.get("certificate_sha256") == sha256_hex(body),
        "certificate_sha256 mismatch (certificate body altered)")

    pm = inst["power_model"]
    H = inst["horizon_us"]
    n_embd = inst["atlas_provenance"]["n_embd"]
    act_bound = inst.get("activation_mem_bound_bytes")
    reqs = {r["id"]: r for r in inst["requests"]}
    e2e = {p: inst["phones"][p]["e2e_us"] for p in inst["phones"]}

    sched = cert.get("schedule", [])
    seen = {}
    for s in sched:
        rid = s["request"]
        req(rid in reqs, f"schedule references unknown request {rid}")
        req(rid not in seen, f"duplicate/stale completion for request {rid}")
        seen[rid] = s
    # every offered request has exactly one terminal outcome (none omitted)
    for rid in reqs:
        req(rid in seen, f"request {rid} omitted from schedule (no terminal outcome)")

    server_ops = []   # (start, finish, tag)
    phone_ops = {p: [] for p in inst["phones"]}
    server_busy = 0

    # per-batch validation
    batches = {b["batch_id"]: b for b in cert.get("server_batches", [])}
    mid_batch = {}
    for b in cert.get("server_batches", []):
        members = b["members"]
        req(len(set(members)) == len(members), f"batch {b['batch_id']} has duplicate members")
        wsset = {reqs[m]["weight_set"] for m in members if m in reqs}
        waveset = {reqs[m]["wave"] for m in members if m in reqs}
        req(len(wsset) == 1 and next(iter(wsset)) == b["weight_set"],
            f"batch {b['batch_id']} members not one weight_set")
        req(len(waveset) == 1, f"batch {b['batch_id']} mixes waves")
        toks = sum(reqs[m]["mid_tokens"] for m in members if m in reqs)
        req(toks == b["tokens"], f"batch {b['batch_id']} tokens {b['tokens']} != sum {toks}")
        req(b["latency_us"] == lffn(inst, toks),
            f"batch {b['batch_id']} latency {b['latency_us']} != L_ffn({toks})={lffn(inst,toks)}")
        req(b["finish"] == b["start"] + b["latency_us"], f"batch {b['batch_id']} finish != start+lat")
        server_ops.append((b["start"], b["finish"], f"batch{b['batch_id']}"))
        server_busy += b["latency_us"]
        for m in members:
            req(m not in mid_batch, f"request {m} appears in two batches")
            mid_batch[m] = b["batch_id"]

    # per-request node checks
    tardy = timeout = met = 0
    intervals_by_device = {}  # device -> list of (start,finish) for activation-mem peak
    for rid, s in seen.items():
        r = reqs[rid]
        pre_us = r.get("pre_us", inst["pre_us"])
        suf_us = r.get("suf_us", inst["suf_us"])
        # PRE
        req(s["pre_start"] >= r["release_us"], f"{rid}: PRE starts before release")
        req(s["pre_finish"] == s["pre_start"] + pre_us, f"{rid}: PRE latency wrong")
        server_ops.append((s["pre_start"], s["pre_finish"], f"pre_{rid}"))
        server_busy += pre_us
        # MID
        dev = s["mid_device"]
        req(s["mid_start"] >= s["pre_finish"], f"{rid}: MID starts before PRE done (not READY)")
        if dev == "SERVER":
            req(rid in mid_batch, f"{rid}: server MID not in any batch")
            b = batches[mid_batch[rid]]
            req(s["mid_start"] == b["start"] and s["mid_finish"] == b["finish"],
                f"{rid}: MID times != its batch times")
        else:
            req(dev in inst["phones"], f"{rid}: unknown phone {dev}")
            resident = r["weight_set"] in inst["phones"][dev]["resident_weight_sets"]
            certified = r["mid_tokens"] in inst["phones"][dev]["certified_tokens"]
            req(resident and certified, f"{rid}: illegal phone route {dev} (residency/certified)")
            req(s["mid_finish"] == s["mid_start"] + e2e[dev],
                f"{rid}: phone MID latency != measured e2e {e2e[dev]}")
            req(rid in cert.get("phone_islands", {}).get(dev, []),
                f"{rid}: on {dev} but not listed in phone_islands")
            phone_ops[dev].append((s["mid_start"], s["mid_finish"], rid))
        intervals_by_device.setdefault(dev, []).append((s["mid_start"], s["mid_finish"]))
        # SUF (D2H-before-credit: suffix cannot start before MID result returns)
        req(s["suf_start"] >= s["mid_finish"], f"{rid}: SUF starts before MID result (D2H) done")
        req(s["suf_finish"] == s["suf_start"] + suf_us, f"{rid}: SUF latency wrong")
        if suf_us > 0:
            server_ops.append((s["suf_start"], s["suf_finish"], f"suf_{rid}"))
            server_busy += suf_us
        # outcome
        completed = s["suf_finish"]
        if completed > H:
            expect = "TIMEOUT"; timeout += 1
        elif completed <= r["deadline_us"]:
            expect = "MET"; met += 1
        else:
            expect = "TARDY"; tardy += 1
        req(s["outcome"] == expect,
            f"{rid}: outcome {s['outcome']} but recomputed {expect} (finish={completed}, "
            f"deadline={r['deadline_us']}, horizon={H})")
        req(s["deadline"] == r["deadline_us"], f"{rid}: deadline field tampered")

    # server single machine: no overlap
    server_ops.sort()
    for i in range(len(server_ops) - 1):
        req(server_ops[i][1] <= server_ops[i + 1][0],
            f"server ops overlap: {server_ops[i][2]} and {server_ops[i+1][2]}")
    # phone lanes: no overlap
    for p in phone_ops:
        ops = sorted(phone_ops[p])
        for i in range(len(ops) - 1):
            req(ops[i][1] <= ops[i + 1][0], f"phone {p} islands overlap: {ops[i][2]},{ops[i+1][2]}")

    # activation-memory peak per device
    if act_bound is not None:
        for dev, ivs in intervals_by_device.items():
            pts = sorted({t for iv in ivs for t in iv})
            peak = 0
            for t in pts:
                cur = sum(r_ for (a, b) in ivs if a <= t < b
                          for r_ in [n_embd * 4 * 16])  # each island holds M*n_embd*4 (M=16)
                peak = max(peak, cur)
            req(peak <= act_bound, f"activation memory peak {peak} on {dev} exceeds bound {act_bound}")

    # ---- energy recomputation ----
    phone_busy = {p: sum((b - a) for (a, b, _) in phone_ops[p]) for p in phone_ops}
    e_server = pm["gpu_active_mw"] * server_busy
    if pm.get("count_server_idle", False):
        e_server += pm["gpu_idle_mw"] * max(0, H - server_busy)
    e_phone = 0
    for p in phone_ops:
        e_phone += (pm["phone_mw"][p] + pm["usb_host_mw"]) * phone_busy[p]
        if pm.get("count_phone_idle", False):
            e_phone += pm.get("phone_idle_mw", 0) * max(0, H - phone_busy[p])
    total = e_server + e_phone
    req(cert.get("energy_server_nj") == e_server,
        f"server energy {cert.get('energy_server_nj')} != recomputed {e_server}")
    req(cert.get("energy_phone_nj") == e_phone,
        f"phone energy {cert.get('energy_phone_nj')} != recomputed {e_phone}")
    req(cert.get("energy_nj") == total, f"total energy {cert.get('energy_nj')} != recomputed {total}")
    req(cert.get("tardy") == tardy and cert.get("timeout") == timeout and cert.get("met") == met,
        f"outcome counts mismatch: cert(t={cert.get('tardy')},to={cert.get('timeout')},"
        f"m={cert.get('met')}) recomputed(t={tardy},to={timeout},m={met})")

    # ---- HBM relief rule (mirrored earns no HBM credit) ----
    hbm = cert.get("hbm", {"mode": {}, "relief_bytes": 0})
    relief_claim = hbm.get("relief_bytes", 0)
    legit_relief = 0
    wbytes = inst["atlas_provenance"]["weight_bytes_f16"]
    for ws, mode in hbm.get("mode", {}).items():
        if mode == "exclusive":
            legit_relief += wbytes
        # mirrored / server => 0 HBM credit
    req(relief_claim <= legit_relief,
        f"HBM relief {relief_claim} exceeds legitimate exclusive-only relief {legit_relief} "
        f"(mirrored/server weights earn no HBM credit)")

    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--certificate", required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    try:
        inst = json.load(open(args.instance))
        cert = json.load(open(args.certificate))
    except Exception as e:
        print(f"FAIL: cannot load inputs: {e}", file=sys.stderr)
        return 3
    fails = check(inst, cert)
    if fails:
        for f in fails:
            print(f"FAIL: {f}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(json.dumps({"valid": True, "instance": inst["instance_id"],
                          "energy_nj": cert["energy_nj"], "met": cert["met"],
                          "tardy": cert["tardy"], "timeout": cert["timeout"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
