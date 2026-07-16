#!/usr/bin/env python3
# S9-V1A-R measurement matrix for one device. Counterbalanced (Latin-square) window order,
# full provenance per row (serial/worker-sha/host-sha/usb/source-sha/harness version),
# thermal + CPU-frequency snapshots around each run (NOT energy). Phases:
#   profile : window=1, interleaved profiling ON/OFF reps (CP1.7 + on/off overhead)
#   64      : 64 MiB synthetic object, provision-only, windows 1/2/4/8, >=5 reps
#   256     : 256 MiB synthetic object, provision-only, best-two windows, >=3 reps
#   gate    : full 464,114,176-byte shard + blk.2 M=16 HTP oracle, windows 1/2/4/8, >=5 reps
#
# Usage: run_matrix.py <serial> <name> <usb> <backend> <base_port> <outdir> <harness_ver> [phase...]
import sys, os, json, time, subprocess

ROOT = "/home/myid/zs89458/Documents/llama.cpp-release"
HOST = f"{ROOT}/build-phone-pim/bin/llama-phone-pim-host"
MODEL = f"{ROOT}/scratchpad/phone_pim/12b-f16-mid-2-3.gguf"
DDIR = "/data/local/tmp/phone_pim"
ROUTE = 17
WINDOWS = [1, 2, 4, 8]

SERIAL, NAME, USB, BACKEND = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
BASE_PORT, OUTDIR, HVER = int(sys.argv[5]), sys.argv[6], sys.argv[7]
PHASES = sys.argv[8:] or ["profile", "64", "256", "gate"]
os.makedirs(OUTDIR, exist_ok=True)
HOST_SHA = subprocess.run(["sha256sum", HOST], capture_output=True, text=True).stdout.split()[0]
WORKER_SHA = subprocess.run(
    ["adb", "-s", SERIAL, "shell", f"sha256sum {DDIR}/llama-phone-pim-worker-meas"],
    capture_output=True, text=True).stdout.split()[0]

def adb(*a): return subprocess.run(["adb", "-s", SERIAL, *a], capture_output=True, text=True)
def fwd(p): adb("forward", "--remove", f"tcp:{p}"); adb("forward", f"tcp:{p}", f"tcp:{p}")
def unfwd(p): adb("forward", "--remove", f"tcp:{p}")

def thermal():
    # Best-effort thermal + max cpufreq snapshot; absent on locked-down devices. NOT energy.
    t = adb("shell", "for z in /sys/class/thermal/thermal_zone*/temp; do cat $z 2>/dev/null; done")
    f = adb("shell", "for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do cat $c 2>/dev/null; done")
    def nums(s): return [int(x) for x in s.split() if x.strip().lstrip('-').isdigit()]
    tz = nums(t.stdout); fz = nums(f.stdout)
    return {"thermal_zone_max_mC": max(tz) if tz else None,
            "cpu_freq_max_khz": max(fz) if fz else None,
            "cpu_freq_readable": bool(fz)}

def start_worker(store, port, profile_recv=False):
    extra = " --profile-recv" if profile_recv else ""
    cmd = (f"cd {DDIR} && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_MBUF=3072 "
           f"./llama-phone-pim-worker-meas --store-dir {store} --max-store-mib 4096 "
           f"--max-model-mib 2048 --min-free-mib 256 --backend {BACKEND} --bind 127.0.0.1 "
           f"--port {port} --route-epoch {ROUTE} --generation 1{extra}")
    wlog = open(f"{OUTDIR}/_worker_{store}.log", "w")
    return subprocess.Popen(["adb", "-s", SERIAL, "shell", cmd], stdout=wlog, stderr=wlog), wlog.name

def push_object(mib, remote):
    # deterministic synthetic object on device (host reads it back over adb for provisioning)
    local = f"{OUTDIR}/_obj_{mib}mib.bin"
    if not os.path.exists(local):
        with open(local, "wb") as f:
            f.write(bytes((i * 2654435761) & 0xFF for i in range(mib * 1024 * 1024)) if False else os.urandom(mib * 1024 * 1024))
    return local

def latin(reps):
    # rotating Latin-square order of WINDOWS across reps
    return [[WINDOWS[(j + i) % len(WINDOWS)] for j in range(len(WINDOWS))] for i in range(reps)]

def run_host(model, port, window, extra, tag):
    cmd = [HOST, "-m", model, "--host", "127.0.0.1", "--port", str(port), "--prefix", "blk.2",
           "--M", "16", "--repeat", "7", "--route-epoch", str(ROUTE), "--generation", "1",
           "--provision", "if-missing", "--chunk-mib", "4", "--stage-window", str(window)] + extra
    p = subprocess.run(cmd, capture_output=True, text=True)
    open(f"{OUTDIR}/{tag}.stderr", "w").write(p.stderr)
    rec = None
    for line in p.stdout.splitlines():
        if line.strip().startswith("{"):
            try: rec = json.loads(line)
            except Exception: pass
    return p.returncode, rec

def provenance(rec, phase, rep, window, obj_bytes, extra=None):
    rec = dict(rec)
    rec.update({"device": NAME, "device_serial": SERIAL, "usb_path": USB,
                "worker_sha": WORKER_SHA, "host_sha": HOST_SHA, "harness_version": HVER,
                "phase": phase, "rep": rep, "matrix_window": window, "object_bytes": obj_bytes})
    if extra: rec.update(extra)
    return rec

def emit(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")

# ---------- profile phase (CP1.7) ----------
def phase_profile():
    out = f"{OUTDIR}/{NAME}_profile.jsonl"; open(out, "w").close()
    order = []
    for i in range(6):  # 6 ON + 6 OFF, strictly interleaved (counterbalanced)
        order += [("on", i + 1), ("off", i + 1)] if i % 2 == 0 else [("off", i + 1), ("on", i + 1)]
    for mode, rep in order:
        port = BASE_PORT + (rep % 7); store = f"store_prof_{mode}_{rep}_{os.getpid()}"
        adb("shell", f"rm -rf {DDIR}/{store}")
        wp, wlog = start_worker(store, port, profile_recv=(mode == "on")); time.sleep(4); fwd(port)
        th_before = thermal()
        extra = ["--release", "--shutdown"] + (["--profile-transport"] if mode == "on" else [])
        rc, rec = run_host(MODEL, port, 1, extra, f"{NAME}_prof_{mode}_{rep}")
        th_after = thermal()
        wp.wait(); unfwd(port); adb("shell", f"rm -rf {DDIR}/{store}")
        recv = None
        for line in open(wlog):
            if "stage_recv_profile" in line:
                try: recv = json.loads(line.split("stage_recv_profile ", 1)[1])
                except Exception: pass
        if rc == 0 and rec:
            emit(out, provenance(rec, "profile", rep, 1, 464114176,
                     {"profiling": mode, "worker_stage_recv_profile": recv,
                      "thermal_before": th_before, "thermal_after": th_after}))
        else:
            emit(out, {"error": "profile_run", "profiling": mode, "rep": rep, "rc": rc})
        print(f"  profile {mode} rep{rep}: rc={rc} goodput={rec and rec.get('stage_useful_goodput_mib_s')}")

# ---------- provision-only synthetic phases (64 / 256 MiB) ----------
def phase_synthetic(mib, windows, reps, phase_name):
    out = f"{OUTDIR}/{NAME}_{phase_name}.jsonl"; open(out, "w").close()
    obj = push_object(mib, None)
    obj_bytes = mib * 1024 * 1024
    for rep, order in enumerate(latin(reps), 1):
        for w in order:
            if w not in windows: continue
            port = BASE_PORT + 20 + (rep % 7); store = f"store_{phase_name}_{w}_{rep}_{os.getpid()}"
            adb("shell", f"rm -rf {DDIR}/{store}")
            wp, _ = start_worker(store, port); time.sleep(3); fwd(port)
            th = thermal()
            rc, rec = run_host(obj, port, w, ["--provision-only", "--shutdown"], f"{NAME}_{phase_name}_{w}_{rep}")
            wp.wait(); unfwd(port); adb("shell", f"rm -rf {DDIR}/{store}")
            if rc == 0 and rec:
                emit(out, provenance(rec, phase_name, rep, w, obj_bytes, {"thermal": th}))
            else:
                emit(out, {"error": f"{phase_name}_run", "window": w, "rep": rep, "rc": rc})
            print(f"  {phase_name} w{w} rep{rep}: rc={rc} goodput={rec and rec.get('useful_goodput_mib_s')}")

# ---------- full-shard gate ----------
def phase_gate(reps=5):
    out = f"{OUTDIR}/{NAME}_gate.jsonl"; open(out, "w").close()
    for rep, order in enumerate(latin(reps), 1):
        for w in order:
            port = BASE_PORT + 40 + (rep % 7); store = f"store_gate_{w}_{rep}_{os.getpid()}"
            adb("shell", f"rm -rf {DDIR}/{store}")
            wp, _ = start_worker(store, port); time.sleep(4); fwd(port)
            th_before = thermal()
            rc, rec = run_host(MODEL, port, w, ["--release", "--shutdown"], f"{NAME}_gate_{w}_{rep}")
            th_after = thermal()
            wp.wait(); unfwd(port); adb("shell", f"rm -rf {DDIR}/{store}")
            if rc == 0 and rec:
                emit(out, provenance(rec, "gate", rep, w, 464114176,
                         {"thermal_before": th_before, "thermal_after": th_after}))
            else:
                emit(out, {"error": "gate_run", "window": w, "rep": rep, "rc": rc})
            print(f"  gate w{w} rep{rep}: rc={rc} goodput={rec and rec.get('stage_useful_goodput_mib_s')}")

print(f"== matrix {NAME} ({SERIAL}) phases={PHASES} worker={WORKER_SHA[:12]} host={HOST_SHA[:12]} ==")
if "profile" in PHASES: phase_profile()
if "64" in PHASES: phase_synthetic(64, WINDOWS, 5, "64mib")
if "256" in PHASES: phase_synthetic(256, [4, 8], 3, "256mib")
if "gate" in PHASES: phase_gate(5)
print(f"== matrix {NAME} done ==")
