"""Summarise q32_tpu_probe runs: latency (p50), vendor hardware time, TFLOPS, rel-L2 vs fp32 reference, memory.

usage: analyze.py RUNS_ROOT VECTORS_DIR OUT.json TAG[=REF_KEY[:INVOKES_PER_PASS]] ...
  TAG          run tag (log RUNS_ROOT/*/TAG.log, outputs out.TAG.case<c>.f32)
  REF_KEY      reference/input key (default M<batch> parsed from the tag), e.g. M256h
  INVOKES_PER_PASS  invokes needed for one 512-token pass (e.g. 2 for M256h), only for reporting
FLOPs per layer pass = 6 * M * 5120 * 25600 (gate, up, down GEMMs; activations ignored).
"""
import glob
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

E, F = 5120, 25600


def find(root, name):
    hits = sorted(glob.glob(f"{root}/*/{name}"))
    return Path(hits[-1]) if hits else None


def parse_log(text):
    calls = [dict(phase=m[0], invoke=int(m[1]) / 1e6, worker=int(m[2]) / 1e6, repeat=m[3] == "1")
             for m in re.findall(r"call=\d+ phase=(\w+) case=\d+ invoke_ns=(\d+) worker_ns=(\d+) finite=\d repeat_exact=(\d)", text)]
    out = {"calls": calls}
    m = re.search(r"metric name=hardware_execution_time_us type=\d value=(\d+)", text)
    n = re.search(r"metric name=number_of_graph_executions type=\d value=(\d+)", text)
    if m and n and int(n.group(1)):
        out["hw_ms"] = int(m.group(1)) / int(n.group(1)) / 1e3
    m = re.search(r"load env_ns=(\d+) model_file_ns=(\d+) compiled_model_ns=(\d+) buffers_ns=(\d+)", text)
    if m:
        out["load_ms"] = {k: int(v) / 1e6 for k, v in zip(("env", "model_file", "compiled_model", "buffers"), m.groups())}
    mem = {}
    for tag, body in re.findall(r"mem tag=(\w+) (.*)", text):
        kv = dict(re.findall(r"(\w+)=(\d+)", body))
        mem[tag] = {k: int(v) for k, v in kv.items() if k in ("VmRSS", "RssFile", "RssAnon", "MemAvailable", "dmabuf_bytes")}
        big = re.search(r"big=(.*)", body)
        mem[tag]["dmabuf_big"] = big.group(1).replace("\t", " ") if big else ""
    out["mem"] = mem
    out["pass"] = "normal_exit=1" in text
    out["dispatch"] = "Replacing 1 out of 1 node(s) with delegate (DispatchDelegate)" in text and "accelerator=npu_only" in text
    w = re.search(r"timed_window_ns=(\d+) calls=(\d+)", text)
    if w:
        out["timed_window_ms"] = int(w.group(1)) / 1e6
    return out


def main():
    root, vec, out_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    rows = []
    for spec in sys.argv[4:]:
        tag, _, rest = spec.partition("=")
        ref_key, _, per_pass = rest.partition(":")
        if not ref_key:
            ref_key = "M" + re.search(r"-M(\d+)", tag).group(1)
        per_pass = int(per_pass or 1)
        m = int(re.match(r"M(\d+)", ref_key).group(1))
        log = find(root, f"{tag}.log")
        if log is None:
            print("missing", tag)
            continue
        info = parse_log(log.read_text())
        timed = [c for c in info["calls"] if c["phase"] == "timed"]
        rec = {"tag": tag, "M": m, "ref": ref_key, "invokes_per_512": per_pass, "pass": info["pass"], "dispatch": info["dispatch"],
               "n_timed": len(timed), "repeat_exact": all(c["repeat"] for c in info["calls"]),
               "first_invoke_ms": next((c["invoke"] for c in info["calls"] if c["phase"] == "first"), None),
               "load_ms": info.get("load_ms"), "mem": info["mem"], "hw_ms": info.get("hw_ms")}
        if timed:
            inv = [c["invoke"] for c in timed]
            wk = [c["worker"] for c in timed]
            rec.update(invoke_p50_ms=statistics.median(inv), invoke_min_ms=min(inv), invoke_max_ms=max(inv),
                       worker_p50_ms=statistics.median(wk))
            flops = 6.0 * m * E * F
            rec["tflops_invoke"] = flops / (rec["invoke_p50_ms"] / 1e3) / 1e12
            if rec["hw_ms"]:
                rec["tflops_hw"] = flops / (rec["hw_ms"] / 1e3) / 1e12
            rec["ms_per_512_tokens_invoke"] = rec["invoke_p50_ms"] * 512 / m
        cases = sorted(log.parent.glob(f"out.{tag}.case*.f32"))
        ref_file = vec / f"ref_{ref_key}.f32"
        if cases and ref_file.exists():
            ref = np.fromfile(ref_file, dtype="<f4").reshape(-1, m, E)
            rels, rowmax = [], []
            for c, path in enumerate(cases):
                y = np.fromfile(path, dtype="<f4").reshape(m, E).astype(np.float64)
                r = ref[c].astype(np.float64)
                rels.append(float(np.linalg.norm(y - r) / np.linalg.norm(r)))
                rowmax.append(float(np.max(np.linalg.norm(y - r, axis=1) / np.linalg.norm(r, axis=1))))
            rec["rel_l2"] = max(rels)
            rec["rel_l2_cases"] = rels
            rec["max_row_rel_l2"] = max(rowmax)
            rec["finite"] = bool(all(np.isfinite(np.fromfile(p, dtype="<f4")).all() for p in cases))
        rows.append(rec)
        print(f"{tag:28s} M={m:4d} invoke_p50={rec.get('invoke_p50_ms', float('nan')):9.3f} hw={rec['hw_ms'] or float('nan'):9.3f} "
              f"TF(inv)={rec.get('tflops_invoke', float('nan')):6.3f} TF(hw)={rec.get('tflops_hw', float('nan')):6.3f} "
              f"rel={rec.get('rel_l2', float('nan')):.3e} worker={rec.get('worker_p50_ms', float('nan')):8.2f} "
              f"dmabuf={rec['mem'].get('after_first_invoke', {}).get('dmabuf_bytes', 0) / 2**20:7.1f}MiB")
    out_path.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
