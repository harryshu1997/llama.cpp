#!/usr/bin/env python3
# S9-V1A-R repaired adversarial harness. Structural JSON parsing (no grep), exits
# nonzero if ANY test fails, persists every host stdout/stderr/exit-status and recovery
# record, and computes retry/waste from the persisted first-run and resumed-run records.
#
# T1 windowed resume-after-partial   T2 windowed kill-mid-flight   T3 durable-result identity
#
# Usage: adversarial_tests.py <serial> <backend> <base_port> <outdir>
import sys, os, json, time, subprocess, signal

ROOT = "/home/myid/zs89458/Documents/llama.cpp-release"
HOST = f"{ROOT}/build-phone-pim/bin/llama-phone-pim-host"
MODEL = f"{ROOT}/scratchpad/phone_pim/12b-f16-mid-2-3.gguf"
DDIR = "/data/local/tmp/phone_pim"
OBJECT_BYTES = 464114176
CHUNK = 4194304
MODEL_SHA = "5cfba18d2a47acc190f317d650895bcc53e914e9a0bc61631860be9591ed360d"
ROUTE = 17

SERIAL, BACKEND, BASE_PORT, OUTDIR = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
os.makedirs(OUTDIR, exist_ok=True)

def adb(*args, **kw):
    return subprocess.run(["adb", "-s", SERIAL, *args], capture_output=True, text=True, **kw)

def start_worker(store, port):
    cmd = (f"cd {DDIR} && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_MBUF=3072 "
           f"./llama-phone-pim-worker-meas --store-dir {store} --max-store-mib 2048 "
           f"--max-model-mib 1024 --min-free-mib 256 --backend {BACKEND} --bind 127.0.0.1 "
           f"--port {port} --route-epoch {ROUTE} --generation 1")
    return subprocess.Popen(["adb", "-s", SERIAL, "shell", cmd],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def kill_worker():
    adb("shell", "pkill -9 -f llama-phone-pim-worker-meas")

def fwd(port):
    adb("forward", "--remove", f"tcp:{port}")
    adb("forward", f"tcp:{port}", f"tcp:{port}")

def unfwd(port):
    adb("forward", "--remove", f"tcp:{port}")

def host_run(port, window, extra, tag):
    """Run host, persist stdout/stderr/exit, return (exit, last_json_record_or_None)."""
    cmd = [HOST, "-m", MODEL, "--host", "127.0.0.1", "--port", str(port),
           "--prefix", "blk.2", "--M", "16", "--repeat", "7", "--route-epoch", str(ROUTE),
           "--generation", "1", "--provision", "if-missing", "--chunk-mib", "4",
           "--stage-window", str(window)] + extra
    p = subprocess.run(cmd, capture_output=True, text=True)
    open(f"{OUTDIR}/{tag}.stdout", "w").write(p.stdout)
    open(f"{OUTDIR}/{tag}.stderr", "w").write(p.stderr)
    open(f"{OUTDIR}/{tag}.exit", "w").write(str(p.returncode))
    rec = None
    for line in p.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rec = json.loads(line)
            except Exception:
                pass
    return p.returncode, rec

results = []
def record(name, ok, detail):
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

# ---------------- T1: windowed resume-after-partial ----------------
def t1():
    port = BASE_PORT + 1; store = f"store_adv_t1_{os.getpid()}"
    adb("shell", f"rm -rf {DDIR}/{store}")
    unfwd(port); wp = start_worker(store, port); time.sleep(4); fwd(port)
    rc1, r1 = host_run(port, 4, ["--test-stop-after-chunks", "40", "--provision-only"], "t1_partial")
    rc2, r2 = host_run(port, 4, ["--provision-only"], "t1_resume")
    wp.terminate(); adb("shell", f"rm -rf {DDIR}/{store}"); unfwd(port)
    ok = True; why = []
    if not (r1 and r1.get("verdict") == "DYNAMIC_PROVISION_PARTIAL" and r1.get("chunks_sent") == 40):
        ok = False; why.append(f"partial bad: {r1 and r1.get('verdict')},cs={r1 and r1.get('chunks_sent')}")
    if not (rc2 == 0 and r2 and r2.get("verdict") == "DYNAMIC_PROVISION_PASS"
            and r2.get("resume_offset") == 40 * CHUNK and r2.get("model_sha256") == MODEL_SHA
            and r2.get("model_source") == "published_store"):
        ok = False; why.append(f"resume bad: rc={rc2},{r2 and r2.get('verdict')},off={r2 and r2.get('resume_offset')}")
    record("T1 windowed resume-after-partial", ok, "; ".join(why) or f"resume_offset={r2.get('resume_offset')}")

# ---------------- T2: windowed kill-mid-flight ----------------
def t2():
    port = BASE_PORT + 2; store = f"store_adv_t2_{os.getpid()}"
    adb("shell", f"rm -rf {DDIR}/{store}")
    unfwd(port); wp = start_worker(store, port); time.sleep(4); fwd(port)
    # provision in background, kill worker mid-flight
    cmd = [HOST, "-m", MODEL, "--host", "127.0.0.1", "--port", str(port), "--prefix", "blk.2",
           "--M", "16", "--repeat", "7", "--route-epoch", str(ROUTE), "--generation", "1",
           "--provision", "if-missing", "--chunk-mib", "4", "--stage-window", "8", "--provision-only"]
    hp = subprocess.Popen(cmd, stdout=open(f"{OUTDIR}/t2_first.stdout", "w"),
                          stderr=open(f"{OUTDIR}/t2_first.stderr", "w"))
    time.sleep(6)
    kill_worker()
    rc1 = hp.wait()
    open(f"{OUTDIR}/t2_first.exit", "w").write(str(rc1))
    wp.wait()
    r1 = None
    for line in open(f"{OUTDIR}/t2_first.stdout").read().splitlines():
        if line.strip().startswith("{"):
            try: r1 = json.loads(line)
            except Exception: pass
    # restart worker on same store, resume to completion
    wp2 = start_worker(store, port); time.sleep(4); fwd(port)
    rc2, r2 = host_run(port, 8, ["--provision-only"], "t2_resume")
    wp2.terminate(); adb("shell", f"rm -rf {DDIR}/{store}"); unfwd(port)
    ok = True; why = []
    if not (rc1 != 0):
        ok = False; why.append(f"first host did not fail (rc={rc1})")
    if not (r1 and r1.get("verdict") == "FAIL_PROVISION"):
        ok = False; why.append("no FAIL_PROVISION record")
    off = r2.get("resume_offset") if r2 else None
    if not (rc2 == 0 and r2 and r2.get("verdict") == "DYNAMIC_PROVISION_PASS"
            and r2.get("model_sha256") == MODEL_SHA and off is not None
            and 0 < off < OBJECT_BYTES and off % CHUNK == 0):
        ok = False; why.append(f"resume bad: rc={rc2},off={off}")
    # retry/waste from persisted records
    retrywaste = {}
    if r1:
        w1_written = r1.get("socket_written_bytes", 0); w1_durable = r1.get("acked_durable_bytes", 0)
        waste = max(0, w1_written - w1_durable)
        retrywaste = {"first_attempted": r1.get("attempted_bytes"), "first_written": w1_written,
                      "first_durable": w1_durable, "waste_bytes": waste,
                      "resume_offset": off, "retry_bytes": waste,
                      "resumed_attempted": r2.get("attempted_bytes") if r2 else None}
        json.dump(retrywaste, open(f"{OUTDIR}/t2_retry_waste.json", "w"), indent=2)
    record("T2 windowed kill-mid-flight", ok,
           "; ".join(why) or f"resume_offset={off} waste={retrywaste.get('waste_bytes')}")

# ---------------- T3: durable-result identity (window=8, full oracle) ----------------
def t3():
    port = BASE_PORT + 3; store = f"store_adv_t3_{os.getpid()}"
    adb("shell", f"rm -rf {DDIR}/{store}")
    unfwd(port); wp = start_worker(store, port); time.sleep(4); fwd(port)
    rc, r = host_run(port, 8, ["--release", "--shutdown"], "t3_full")
    wp.wait(); adb("shell", f"rm -rf {DDIR}/{store}"); unfwd(port)
    ok = (rc == 0 and r is not None
          and r.get("verdict") == "DYNAMIC_FFN_PASS"
          and r.get("model_source") == "published_store"
          and r.get("model_sha256") == MODEL_SHA
          and r.get("stage_bytes_sent") == OBJECT_BYTES
          and r.get("stage_chunk_count") == 111
          and r.get("stage_remote_duplicate_chunks") == 0
          and r.get("stage_remote_accepted_chunks") == 111
          and isinstance(r.get("rel_l2_max"), (int, float)) and r["rel_l2_max"] < 5e-3)
    record("T3 durable-result identity (window=8)", ok,
           f"verdict={r and r.get('verdict')} rel_l2={r and r.get('rel_l2_max')} dup={r and r.get('stage_remote_duplicate_chunks')}")

print(f"== adversarial windowing tests on {SERIAL} ==")
t1(); t2(); t3()
nfail = sum(1 for _, ok, _ in results if not ok)
json.dump([{"name": n, "pass": ok, "detail": d} for n, ok, d in results],
          open(f"{OUTDIR}/summary.json", "w"), indent=2)
print(f"== {SERIAL}: {len(results)-nfail}/{len(results)} PASS ==")
sys.exit(1 if nfail else 0)
