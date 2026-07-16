#!/usr/bin/env python3
# CP4.3: run BOTH phones' full-shard provisioning at their selected window concurrently
# on their separate USB buses. Emits one provenance row per phone to <outdir>/simultaneous.jsonl.
# Usage: run_simultaneous.py <window> <outdir> <harness_ver> <reps>
import sys, os, json, time, subprocess, threading

ROOT = "/home/myid/zs89458/Documents/llama.cpp-release"
HOST = f"{ROOT}/build-phone-pim/bin/llama-phone-pim-host"
MODEL = f"{ROOT}/scratchpad/phone_pim/12b-f16-mid-2-3.gguf"
DDIR = "/data/local/tmp/phone_pim"
ROUTE = 17
DEVICES = [("OP12", "5ae7a43d", "6-2"), ("OP15", "3C15AU002CL00000", "8-3")]
WINDOW, OUTDIR, HVER, REPS = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
os.makedirs(OUTDIR, exist_ok=True)
HOST_SHA = subprocess.run(["sha256sum", HOST], capture_output=True, text=True).stdout.split()[0]
out = f"{OUTDIR}/simultaneous.jsonl"; open(out, "w").close()
lock = threading.Lock()

def adb(serial, *a): return subprocess.run(["adb", "-s", serial, *a], capture_output=True, text=True)

def worker_sha(serial):
    return adb(serial, "shell", f"sha256sum {DDIR}/llama-phone-pim-worker-meas").stdout.split()[0]

def one_phone(name, serial, usb, port, rep):
    store = f"store_sim_{rep}_{os.getpid()}"
    adb(serial, "shell", f"rm -rf {DDIR}/{store}")
    adb(serial, "forward", "--remove", f"tcp:{port}")
    cmd = (f"cd {DDIR} && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_MBUF=3072 "
           f"./llama-phone-pim-worker-meas --store-dir {store} --max-store-mib 4096 --max-model-mib 2048 "
           f"--min-free-mib 256 --backend HTP0 --bind 127.0.0.1 --port {port} --route-epoch {ROUTE} --generation 1")
    wp = subprocess.Popen(["adb", "-s", serial, "shell", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(4)
    adb(serial, "forward", f"tcp:{port}", f"tcp:{port}")
    p = subprocess.run([HOST, "-m", MODEL, "--host", "127.0.0.1", "--port", str(port), "--prefix", "blk.2",
                        "--M", "16", "--repeat", "7", "--route-epoch", str(ROUTE), "--generation", "1",
                        "--provision", "if-missing", "--chunk-mib", "4", "--stage-window", str(WINDOW),
                        "--release", "--shutdown"], capture_output=True, text=True)
    wp.wait()
    adb(serial, "forward", "--remove", f"tcp:{port}")
    adb(serial, "shell", f"rm -rf {DDIR}/{store}")
    rec = None
    for line in p.stdout.splitlines():
        if line.strip().startswith("{"):
            try: rec = json.loads(line)
            except Exception: pass
    if p.returncode == 0 and rec:
        rec.update({"device": name, "device_serial": serial, "usb_path": usb, "worker_sha": worker_sha(serial),
                    "host_sha": HOST_SHA, "harness_version": HVER, "phase": "simultaneous",
                    "rep": rep, "matrix_window": WINDOW, "object_bytes": 464114176, "concurrent": True})
        line = json.dumps(rec)
    else:
        line = json.dumps({"error": "simultaneous", "device": name, "rep": rep, "rc": p.returncode})
    with lock:
        with open(out, "a") as f: f.write(line + "\n")
    print(f"  simultaneous {name} rep{rep}: rc={p.returncode} goodput={rec and rec.get('stage_useful_goodput_mib_s')}")

print(f"== simultaneous window={WINDOW} reps={REPS} ==")
for rep in range(1, REPS + 1):
    threads = [threading.Thread(target=one_phone, args=(n, s, u, 45000 + i, rep))
               for i, (n, s, u) in enumerate(DEVICES)]
    for t in threads: t.start()
    for t in threads: t.join()
    time.sleep(3)
print("== simultaneous done ==")
