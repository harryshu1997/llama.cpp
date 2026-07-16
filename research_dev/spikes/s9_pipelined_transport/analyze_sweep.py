#!/usr/bin/env python3
# S9-V1A-R repaired analyzer: FAIL CLOSED. Returns nonzero unless every required
# device/window/repetition is present with zero errors and every row passes the full
# correctness+provenance contract, and the UNROUNDED best-window/window=1 speedup clears
# the gate on both phones. Run --selftest for mutation tests proving each guard fires.
#
# Usage:
#   analyze_sweep.py --gate <op12.jsonl> <op15.jsonl>   # full-shard DYNAMIC_FFN_PASS gate
#   analyze_sweep.py --selftest
import sys, json, math, statistics as st, collections

GATE_RATIO   = 1.20
REL_L2_MAX   = 5e-3
OBJECT_BYTES = 464114176
CHUNK_COUNT  = 111
MODEL_SHA    = "5cfba18d2a47acc190f317d650895bcc53e914e9a0bc61631860be9591ed360d"
WORKER_SHA   = "c284227179b4048feef928f03f03e984cdbdae3bb2e90f25971b2c98cd764f27"  # measurement worker
EXPECT_DEVICES = {"OP12": "5ae7a43d", "OP15": "3C15AU002CL00000"}
REQUIRED_WINDOWS = [1, 2, 4, 8]
MIN_REPS = 5

def load_rows(path):
    rows = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                rows.append({"__parse_error__": f"{path}:{ln}: {e}"})
    return rows

def validate_device(name, serial, rows, worker_sha):
    """Return (per_window_goodputs, errors)."""
    errs = []
    host_rows = []
    for r in rows:
        if "__parse_error__" in r:
            errs.append(r["__parse_error__"]); continue
        if r.get("verdict") == "FAIL_PROVISION" or "error" in r or "sink_err" in r:
            errs.append(f"{name}: error/failure row present: {str(r)[:120]}")
            continue
        if r.get("role") in ("host", "sink", "source"):
            continue  # transport-bench rows are not gate rows
        if "stage_window" not in r or "stage_useful_goodput_mib_s" not in r:
            continue
        host_rows.append(r)
    # every row must pass the full contract
    for r in host_rows:
        w = r.get("stage_window"); rep = r.get("rep")
        tag = f"{name} w{w} rep{rep}"
        if r.get("device_serial") != serial:
            errs.append(f"{tag}: device_serial {r.get('device_serial')} != {serial}")
        if r.get("worker_sha") != worker_sha:
            errs.append(f"{tag}: worker_sha mismatch")
        if r.get("model_sha256") != MODEL_SHA:
            errs.append(f"{tag}: model_sha256 mismatch")
        if r.get("stage_bytes_sent") != OBJECT_BYTES:
            errs.append(f"{tag}: stage_bytes_sent {r.get('stage_bytes_sent')} != {OBJECT_BYTES}")
        if r.get("stage_chunk_count") != CHUNK_COUNT:
            errs.append(f"{tag}: stage_chunk_count {r.get('stage_chunk_count')} != {CHUNK_COUNT}")
        if r.get("verdict") != "DYNAMIC_FFN_PASS":
            errs.append(f"{tag}: verdict {r.get('verdict')} != DYNAMIC_FFN_PASS")
        if r.get("model_source") != "published_store":
            errs.append(f"{tag}: model_source {r.get('model_source')} != published_store")
        rl = r.get("rel_l2_max")
        if not isinstance(rl, (int, float)) or not math.isfinite(rl) or rl >= REL_L2_MAX:
            errs.append(f"{tag}: rel_l2_max {rl} not finite/<{REL_L2_MAX}")
        if r.get("stage_remote_accepted_chunks") != CHUNK_COUNT:
            errs.append(f"{tag}: accepted_chunks {r.get('stage_remote_accepted_chunks')} != {CHUNK_COUNT}")
        if r.get("stage_remote_duplicate_chunks") != 0:
            errs.append(f"{tag}: duplicate_chunks != 0")
        if r.get("stage_wasted_bytes") not in (0, None):
            errs.append(f"{tag}: wasted_bytes != 0")
    # window/rep coverage: each required window has >= MIN_REPS unique reps
    by_w = collections.defaultdict(set)
    good = collections.defaultdict(list)
    for r in host_rows:
        by_w[r.get("stage_window")].add(r.get("rep"))
        good[r.get("stage_window")].append(r["stage_useful_goodput_mib_s"])
    for w in REQUIRED_WINDOWS:
        if w not in by_w:
            errs.append(f"{name}: required window {w} missing")
        elif None in by_w[w] or len(by_w[w]) < MIN_REPS:
            errs.append(f"{name}: window {w} has {len(by_w[w])} unique reps (<{MIN_REPS})")
    return good, errs

def gate(files):
    if len(files) != len(EXPECT_DEVICES):
        print(f"FAIL: expected {len(EXPECT_DEVICES)} device files, got {len(files)}"); return 1
    # map each file to a device by its rows' device_serial
    serial_to_name = {v: k for k, v in EXPECT_DEVICES.items()}
    all_errs = []
    summary = {}
    seen_serials = set()
    for path in files:
        rows = load_rows(path)
        serials = {r.get("device_serial") for r in rows if isinstance(r, dict) and r.get("device_serial")}
        serials.discard(None)
        if len(serials) != 1:
            all_errs.append(f"{path}: rows span serials {serials} (want exactly 1)"); continue
        serial = next(iter(serials))
        if serial not in serial_to_name:
            all_errs.append(f"{path}: unknown device serial {serial}"); continue
        name = serial_to_name[serial]
        seen_serials.add(serial)
        good, errs = validate_device(name, serial, rows, WORKER_SHA)
        all_errs += errs
        if not errs:
            m = {w: st.median(good[w]) for w in good if good[w]}
            base = m.get(1)
            best_w = max(m, key=lambda w: m[w]) if m else None
            # conservative: min(best-window) / max(window=1)
            cons = (min(good[best_w]) / max(good[1])) if base and best_w and good.get(1) else 0.0
            ratio = (m[best_w] / base) if base and best_w else 0.0
            summary[name] = {"median_goodput": {w: round(m[w], 3) for w in m},
                             "best_window": best_w, "median_ratio": round(ratio, 4),
                             "conservative_ratio": round(cons, 4),
                             "gate_median": ratio >= GATE_RATIO,
                             "gate_conservative": cons >= GATE_RATIO}
    missing = set(EXPECT_DEVICES.values()) - seen_serials
    if missing:
        all_errs.append(f"missing required devices: {missing}")
    if all_errs:
        print("GATE FAIL:")
        for e in all_errs[:40]:
            print("  -", e)
        return 1
    ok = all(s["gate_median"] for s in summary.values()) and len(summary) == len(EXPECT_DEVICES)
    for name, s in summary.items():
        print(f"{name}: windows {s['median_goodput']}  best=w{s['best_window']}  "
              f"median_ratio={s['median_ratio']}x  conservative={s['conservative_ratio']}x  "
              f"gate(median>= {GATE_RATIO}): {'PASS' if s['gate_median'] else 'FAIL'}")
    print(f"OVERALL GATE: {'PASS' if ok else 'FAIL'}")
    json.dump(summary, open(sys.argv[-1] + ".summary.json", "w") if False else sys.stdout, indent=2) if False else None
    return 0 if ok else 1

# ---------------- self tests (mutation) ----------------
def _good_row(name, serial, w, rep):
    return {"verdict": "DYNAMIC_FFN_PASS", "model_source": "published_store",
            "device_serial": serial, "worker_sha": WORKER_SHA, "model_sha256": MODEL_SHA,
            "stage_bytes_sent": OBJECT_BYTES, "stage_chunk_count": CHUNK_COUNT,
            "stage_remote_accepted_chunks": CHUNK_COUNT, "stage_remote_duplicate_chunks": 0,
            "stage_wasted_bytes": 0, "rel_l2_max": 2.9e-4, "stage_window": w, "rep": rep,
            "stage_useful_goodput_mib_s": (15.0 if w == 1 else 37.0)}

def _good_set(name, serial):
    return [_good_row(name, serial, w, rep) for w in REQUIRED_WINDOWS for rep in range(1, MIN_REPS + 1)]

def selftest():
    import tempfile, os
    passed = 0; failed = 0
    def check(desc, rows_by_dev, expect_pass):
        nonlocal passed, failed
        paths = []
        for dev, rows in rows_by_dev.items():
            fd, p = tempfile.mkstemp(suffix=".jsonl"); os.close(fd)
            with open(p, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            paths.append(p)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = gate(paths)
        for p in paths:
            os.unlink(p)
        got_pass = (rc == 0)
        good = (got_pass == expect_pass)
        print(f"  [{'ok' if good else 'XX'}] {desc}: rc={rc} expect_pass={expect_pass}")
        if good: passed += 1
        else: failed += 1

    op12, op15 = EXPECT_DEVICES["OP12"], EXPECT_DEVICES["OP15"]
    base = {op12: _good_set("OP12", op12), op15: _good_set("OP15", op15)}
    check("all good -> PASS", base, True)

    def mutate(fn):
        import copy
        d = {k: [dict(r) for r in v] for k, v in base.items()}
        fn(d); return d
    check("missing device -> FAIL", {op12: _good_set("OP12", op12)}, False)
    check("missing window (drop w4) -> FAIL",
          mutate(lambda d: d.__setitem__(op12, [r for r in d[op12] if r["stage_window"] != 4])), False)
    check("too few reps (drop reps of w2) -> FAIL",
          mutate(lambda d: d.__setitem__(op12, [r for r in d[op12] if not (r["stage_window"] == 2 and r["rep"] > 2)])), False)
    check("error row present -> FAIL",
          mutate(lambda d: d[op12].append({"error": "run", "stage_window": 4})), False)
    check("wrong device_serial -> FAIL",
          mutate(lambda d: d[op12][0].__setitem__("device_serial", "deadbeef")), False)
    check("wrong worker_sha -> FAIL",
          mutate(lambda d: d[op12][0].__setitem__("worker_sha", "0"*64)), False)
    check("wrong model_sha -> FAIL",
          mutate(lambda d: d[op12][0].__setitem__("model_sha256", "0"*64)), False)
    check("wrong bytes -> FAIL",
          mutate(lambda d: d[op12][0].__setitem__("stage_bytes_sent", 123)), False)
    check("wrong chunk_count -> FAIL",
          mutate(lambda d: d[op12][0].__setitem__("stage_chunk_count", 99)), False)
    check("verdict not DYNAMIC_FFN_PASS -> FAIL",
          mutate(lambda d: d[op12][0].__setitem__("verdict", "FAIL_CORRECTNESS")), False)
    check("model_source not published_store -> FAIL",
          mutate(lambda d: d[op12][0].__setitem__("model_source", "prestaged")), False)
    check("rel_l2 >= 5e-3 -> FAIL",
          mutate(lambda d: d[op12][0].__setitem__("rel_l2_max", 6e-3)), False)
    check("rel_l2 not finite -> FAIL",
          mutate(lambda d: d[op12][0].__setitem__("rel_l2_max", float("nan"))), False)
    check("accepted_chunks wrong -> FAIL",
          mutate(lambda d: d[op12][0].__setitem__("stage_remote_accepted_chunks", 110)), False)
    check("duplicate_chunks nonzero -> FAIL",
          mutate(lambda d: d[op12][0].__setitem__("stage_remote_duplicate_chunks", 1)), False)
    check("wasted_bytes nonzero -> FAIL",
          mutate(lambda d: d[op12][0].__setitem__("stage_wasted_bytes", 4096)), False)
    check("speedup below gate (all windows equal) -> FAIL",
          mutate(lambda d: [r.__setitem__("stage_useful_goodput_mib_s", 15.0) for r in d[op12]]), False)

    print(f"selftest: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        sys.exit(selftest())
    if len(sys.argv) >= 3 and sys.argv[1] == "--gate":
        sys.exit(gate(sys.argv[2:]))
    print("usage: analyze_sweep.py --gate <op12.jsonl> <op15.jsonl> | --selftest")
    sys.exit(2)
