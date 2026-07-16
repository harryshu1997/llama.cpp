#!/usr/bin/env python3
# CP1.7 honest profiling report from the profile-phase rows. Reports p50/p95/CoV of
# window=1 goodput (profiling ON and OFF) and the profiling on/off overhead, plus the
# repaired stage-scoped accounting. IMPORTANT: host-side and phone-side CPU timers are
# reported SEPARATELY and never summed as a single wall fraction (they run on different
# CPUs and overlap in wall time).
import sys, json, statistics as st

def p(vs, q):
    if not vs: return 0.0
    vs = sorted(vs); k = (len(vs) - 1) * q
    lo = int(k); hi = min(lo + 1, len(vs) - 1)
    return vs[lo] + (vs[hi] - vs[lo]) * (k - lo)
def cov(vs): return (st.pstdev(vs) / st.mean(vs) * 100) if len(vs) > 1 and st.mean(vs) else 0.0

def report(path):
    on, off = [], []
    for line in open(path):
        line = line.strip()
        if not line: continue
        o = json.loads(line)
        if "error" in o: continue
        (on if o.get("profiling") == "on" else off).append(o)
    name = (on or off)[0].get("device") if (on or off) else path
    print(f"\n===== {name} profiling ({path}) =====")
    print(f"  reps: profiling_on={len(on)} profiling_off={len(off)}")
    for label, rows in [("ON", on), ("OFF", off)]:
        g = [r["stage_useful_goodput_mib_s"] for r in rows]
        w = [r["stage_e2e_ms"] for r in rows]
        if g:
            print(f"  [{label}] goodput MiB/s p50={p(g,.5):.2f} p95={p(g,.95):.2f} CoV={cov(g):.1f}%  "
                  f"wall_ms p50={p(w,.5):.0f}")
    if on and off:
        won = st.median([r["stage_e2e_ms"] for r in on]); woff = st.median([r["stage_e2e_ms"] for r in off])
        print(f"  profiling overhead (median wall ON/OFF): {won:.0f}/{woff:.0f} ms = {won/woff:.3f}x")
    if not on:
        print("  (no profiling-on reps; cannot report the repaired accounting)")
        return
    # Repaired accounting from profiling-ON reps (medians). Host timeline is single-CPU serial.
    def med(key): return st.median([r[key] for r in on if r.get(key) is not None])
    wall = med("stage_e2e_ms")
    host = {k: med(f"stage_host_{k}_ms") for k in
            ["read", "hash", "prefix_hash", "send", "frame_sha", "socket_send", "ack_wait"]}
    host_sum = host["read"] + host["hash"] + host["prefix_hash"] + host["send"] + host["ack_wait"]
    print(f"  -- HOST-side serial timeline (single CPU; sums are valid) --  wall={wall:.0f} ms")
    print(f"     read={host['read']:.0f} hash(data)={host['hash']:.0f} prefix_hash(data)={host['prefix_hash']:.0f} "
          f"send={host['send']:.0f} ack_wait={host['ack_wait']:.0f}")
    print(f"     host timeline accounted: {host_sum/wall*100:.2f}% of wall")
    print(f"     host_send splits into frame_sha(envelope+data)={host['frame_sha']:.0f} + "
          f"socket_send={host['socket_send']:.0f}  (send={host['send']:.0f})")
    # Phone-side (inside ack_wait): reported separately, NOT summed with host as a wall fraction.
    recv = [r["worker_stage_recv_profile"] for r in on if r.get("worker_stage_recv_profile")]
    def rmed(k): return st.median([x[k] for x in recv]) if recv else 0.0
    rem = {k: med(f"stage_remote_{k}_ms") for k in ["chunk_hash", "write", "data_sync", "full_verify"]}
    print(f"  -- PHONE-side, occurs INSIDE host_ack_wait (DIFFERENT CPU; do NOT add to host timeline) --")
    print(f"     stage recv_header(idle+socket)={rmed('recv_header_ms'):.0f} recv_payload={rmed('recv_payload_ms'):.0f}")
    print(f"     frame_sha(envelope+data)={rmed('frame_sha_ms'):.0f} store_chunk_hash(data)={rem['chunk_hash']:.0f} "
          f"store_prefix_hash(data)={rmed('store_prefix_hash_ms'):.0f}")
    print(f"     write={rem['write']:.0f} data_sync={rem['data_sync']:.0f} full_verify(whole file, once)={rem['full_verify']:.0f}")
    print(f"  -- hash domains: outer-frame SHA covers 56-byte envelope + data; "
          f"manifest/chunk/prefix SHA covers data only --")
    # thermal band
    tb = [r.get("thermal_before", {}) for r in on if r.get("thermal_before")]
    if tb and tb[0].get("thermal_zone_max_mC") is not None:
        temps = [t["thermal_zone_max_mC"] / 1000.0 for t in tb if t.get("thermal_zone_max_mC")]
        freqs = [t["cpu_freq_max_khz"] / 1000.0 for t in tb if t.get("cpu_freq_max_khz")]
        print(f"  thermal band: zone_max {min(temps):.0f}-{max(temps):.0f} C; "
              f"cpu_freq {'%.0f-%.0f MHz'%(min(freqs),max(freqs)) if freqs else 'unreadable'}")

for path in sys.argv[1:]:
    report(path)
